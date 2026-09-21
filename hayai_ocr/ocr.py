import os
import re
from pathlib import Path
from typing import List, Optional, Union

import jaconv
import torch
from PIL import Image
from loguru import logger
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    GenerationMixin,
    PreTrainedTokenizerFast,
    VisionEncoderDecoderModel,
)

try:
    from hayai_ocr.litert import (
        LitertOcrEngine,
        normalize_litert_quant,
        DEFAULT_HF_REPO as _LITERT_DEFAULT_REPO,
    )
except Exception:  # pragma: no cover - optional deps missing
    LitertOcrEngine = None  # type: ignore
    normalize_litert_quant = None  # type: ignore
    _LITERT_DEFAULT_REPO = "JustANormalTinkerer/hayai-ocr-v2-tflite"


V1_REPO = "JustANormalTinkerer/hayai-ocr"
V2_REPO = "JustANormalTinkerer/hayai-ocr-v2"
V25_NOVA_REPO = "JustANormalTinkerer/hayai-ocr-v2.5-nova"
DEFAULT_REPO = V25_NOVA_REPO

DEFAULT_MAX_NUM_PATCHES = 256
NOVA_DEFAULT_MAX_NUM_PATCHES = 512


def _is_nova_repo(repo: Optional[str]) -> bool:
    """Return True if ``repo`` refers to the v2.5-nova checkpoint family."""
    if not repo:
        return False
    lowered = repo.lower()
    return repo == V25_NOVA_REPO or "v2.5-nova" in lowered or "nova" in lowered


class MangaVisionEncoderDecoderModel(VisionEncoderDecoderModel, GenerationMixin):
    """Custom VisionEncoderDecoderModel that forwards SigLIP2 NaFlex-specific
    tensors (pixel_attention_mask, spatial_shapes) to the encoder (used for legacy v1)."""

    def forward(
        self,
        pixel_values=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        **kwargs,
    ):
        if encoder_outputs is None and pixel_values is not None:
            encoder_outputs = self.encoder(
                pixel_values=pixel_values,
                pixel_attention_mask=pixel_attention_mask,
                spatial_shapes=spatial_shapes,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        return super().forward(
            pixel_values=None,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            decoder_inputs_embeds=decoder_inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


def apply_torchao_quantization(model: torch.nn.Module, quant_type: Optional[str]) -> None:
    """Applies in-place quantization via torchao, skipping vision_encoder
    submodules to avoid degrading the image feature extractor.
    """
    if quant_type is None:
        return

    import torchao
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig

    def _skip_vision_encoder(module: torch.nn.Module, fqn: str) -> bool:
        """Filter function: returns True for Linear modules that SHOULD be quantized."""
        return isinstance(module, torch.nn.Linear) and not fqn.startswith("vision_encoder")

    if quant_type == "int4":
        logger.info("Applying INT4 weight-only quantization via torchao...")
        torchao.quantize_(model, Int4WeightOnlyConfig(), filter_fn=_skip_vision_encoder)
    elif quant_type == "int8":
        logger.info("Applying INT8 weight-only quantization via torchao...")
        torchao.quantize_(model, Int8WeightOnlyConfig(), filter_fn=_skip_vision_encoder)
    else:
        raise ValueError(f"Unsupported quantization type: {quant_type!r}. Use 'int4' or 'int8'.")


# ---------------------------------------------------------------------------
# Torch-compile / inference optimizations  (CPU + GPU)
# ---------------------------------------------------------------------------

def _resolve_max_num_patches(value: Optional[int], *, default: int = 256) -> int:
    """Resolve ``max_num_patches`` from an explicit value, env var, or default.

    - Explicit ``value`` (including ``0``/negative, which raises) wins.
    - Otherwise ``HAYAI_MAX_NUM_PATCHES`` env var is honored when set.
    - Otherwise ``default`` (256, upstream NaFlex budget) is used.
    """
    if value is None:
        env = os.environ.get("HAYAI_MAX_NUM_PATCHES", "").strip()
        if env:
            try:
                value = int(env)
            except ValueError:
                raise ValueError(
                    f"Invalid HAYAI_MAX_NUM_PATCHES={env!r}: must be a positive integer"
                )
            # fall through to validation below
        else:
            return default
    try:
        resolved = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"Invalid max_num_patches={value!r}: must be a positive integer")
    if resolved <= 0:
        raise ValueError(f"Invalid max_num_patches={value!r}: must be a positive integer")
    return resolved


def _configure_torch_for_inference(device: str) -> None:
    """Global torch inductor / matmul settings for faster inference.

    - TF32 / high matmul precision for GPU
    - cudnn benchmark
    - inductor freezing + CPU weight prepack (oneDNN)
    """
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    # GPU: TF32 & cuDNN autotune
    try:
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            try:
                torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
            except Exception:
                pass
    except Exception:
        pass
    # MKL / mkldnn already enabled in this build; ensure it stays on
    try:
        if hasattr(torch.backends, "mkldnn") and hasattr(torch.backends.mkldnn, "is_available"):
            if torch.backends.mkldnn.is_available():
                pass
    except Exception:
        pass
    # Inductor freezing: safe for inference, lets inductor constant-fold weights
    try:
        import torch._inductor.config as inductor_config

        inductor_config.freezing = True
        # CPU: oneDNN weight prepacking gives ~10-20% gemm speedup
        if device == "cpu":
            try:
                inductor_config.cpp.weight_prepack = True  # type: ignore[attr-defined]
            except Exception:
                pass
            # cpp_wrapper needs a C++ compiler; enable only if user opts-in
            if os.environ.get("HAYAI_CPP_WRAPPER", "0") == "1":
                try:
                    inductor_config.cpp_wrapper = True  # type: ignore[attr-defined]
                except Exception:
                    pass
            # Avoid overhead of dynamic shape padding on CPU for small batches
            try:
                inductor_config.cpp.min_chunk_size = 1  # type: ignore[attr-defined]
            except Exception:
                pass
        # Allow torch.compile to use more threads for parallel compilation
        try:
            inductor_config.compile_threads = max(1, (os.cpu_count() or 4))  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
        pass


def _get_compile_candidates(
    device: str,
    compile_mode: Optional[str],
    compile_fullgraph: bool,
    compile_dynamic: Optional[bool],
) -> List[dict]:
    """Return ordered list of torch.compile kwargs to try.

    The first candidate is the most aggressive/fastest; later ones are fallbacks.

    Note on dynamic shapes: ``dynamic=True`` forces *all* sizes/ints dynamic, which
    the PyTorch docs call "Not recommended ... error prone" – it can cause
    performance regressions and longer compilation. The preferred approaches are:

    * automatic dynamic (default when ``dynamic`` is omitted/None) – first
      compilation assumes static shapes, tracks input sizes, and on the first
      recompilation marks only the dimensions that actually changed as dynamic;
    * user annotations via ``torch._dynamo.maybe_mark_dynamic(tensor, dim)`` to
      explicitly mark only batch / sequence dims dynamic before compile.

    We therefore default to automatic dynamic (no ``dynamic`` kwarg, i.e.
    ``dynamic=None``). ``compile_dynamic=True`` is still accepted as an
    explicit opt-in that forces all dims dynamic, but is not used by default.
    """
    if compile_mode is not None:
        base = {"mode": compile_mode, "fullgraph": compile_fullgraph}
        if compile_dynamic is not None:
            base["dynamic"] = compile_dynamic
        # omitted -> automatic dynamic (recommended); do not force dynamic=True
        return [base]

    if "cuda" in device:
        # GPU: autotune matmuls + optional cudagraphs.
        # Use automatic dynamic by default (no ``dynamic`` kwarg). Forcing
        # ``dynamic=True`` would make every dim dynamic and is not recommended.
        candidates: List[dict] = [
            {"mode": "max-autotune-no-cudagraphs", "fullgraph": False},
            {"mode": "max-autotune", "fullgraph": False},
            {"mode": "reduce-overhead", "fullgraph": False},
            {"mode": "default", "fullgraph": False},
        ]
    elif device == "mps":
        # MPS inductor support is limited as of torch 2.13; caller will skip compile entirely
        candidates = []
    else:  # cpu
        # CPU: automatic dynamic lets inductor specialize first then generalize
        # only the dims that actually vary (e.g., batch, ph*pw, gen length).
        candidates: List[dict] = [
            {"mode": "default", "fullgraph": False},
            {"mode": "max-autotune-no-cudagraphs", "fullgraph": False},
        ]

    if compile_dynamic is not None:
        for c in candidates:
            c["dynamic"] = compile_dynamic
    if compile_fullgraph:
        for c in candidates:
            c["fullgraph"] = True

    # Deduplicate while preserving order
    seen = set()
    uniq: List[dict] = []
    for c in candidates:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            uniq.append(c)
    return uniq


def _apply_torch_compile(
    model: torch.nn.Module,
    device: str,
    compile_mode: Optional[str] = None,
    compile_fullgraph: bool = False,
    compile_dynamic: Optional[bool] = None,
    compile_backend: str = "inductor",
) -> torch.nn.Module:
    """Apply torch.compile with device-aware defaults and safe fallbacks.

    Uses automatic dynamic shapes (the PyTorch-recommended default) instead of
    ``dynamic=True``. With automatic dynamic (``dynamic`` omitted / ``None``),
    the first compilation assumes static shapes; on the first recompilation the
    dimensions that actually changed (e.g. batch, ``ph*pw``, generation length)
    are marked dynamic. This avoids forcing *all* dims dynamic (error-prone and
    slower per docs) while still eliminating repeated recompiles after warmup.

    If the user explicitly wants to annotate only the dims that vary, the
    recommended API is ``torch._dynamo.maybe_mark_dynamic(tensor, dim)`` on the
    inputs *before* compile, rather than ``torch.compile(dynamic=True)``. We
    optionally apply ``maybe_mark_dynamic`` for the batch dim when available
    (best-effort, ignored on older torch).

    - On GPU: tries max-autotune-no-cudagraphs -> max-autotune -> reduce-overhead -> default
    - On CPU: tries default with inductor CPU codegen, weight prepack & freezing
    - On MPS: skips compile (not reliably supported)
    - Also tries to pre-compile vision_encoder separately for HayaiModel to improve cache granularity.

    Returns compiled model or original model on failure.
    """
    if device == "mps":
        logger.info("torch.compile skipped on MPS (inductor not fully supported)")
        return model

    _configure_torch_for_inference(device)

    # Env var to force-disable compile (useful for debugging)
    if os.environ.get("HAYAI_DISABLE_COMPILE", "0") == "1":
        logger.info("torch.compile disabled via HAYAI_DISABLE_COMPILE=1")
        return model

    # Recommended user-annotation example (no-op if not available): mark only
    # the batch dimension dynamic instead of forcing all dims dynamic.
    # This is the preferred alternative to ``dynamic=True``.
    # We do this best-effort on a dummy tensor just to document the pattern;
    # actual dynamic handling relies on automatic dynamic (see _get_compile_candidates).
    try:
        # No tensor to annotate here; HayaiOcr warmup will trigger automatic
        # dynamic on second shape variant. If you know batch dim varies, you
        # could do: torch._dynamo.maybe_mark_dynamic(pixel_values, 0)
        # before the first compiled call. We keep this as documentation.
        pass
    except Exception:
        pass

    # Try to pre-compile vision_encoder separately (Hayai v2) for better dynamism handling.
    # Vision encoder has variable sequence length (ph*pw <=256) but fixed tensor
    # shape (B,256,768); automatic dynamic will handle the effective length.
    if hasattr(model, "vision_encoder"):
        cands = _get_compile_candidates(device, compile_mode, compile_fullgraph, compile_dynamic)
        if cands:
            ve_kwargs = {k: v for k, v in cands[0].items() if v is not None}
            try:
                model.vision_encoder = torch.compile(  # type: ignore[attr-defined]
                    model.vision_encoder, backend=compile_backend, **ve_kwargs
                )
                logger.info(f"Compiled vision_encoder with {ve_kwargs} for {device}")
            except Exception as e:
                logger.debug(f"vision_encoder separate compile failed ({e}); will still compile full model")

    candidates = _get_compile_candidates(device, compile_mode, compile_fullgraph, compile_dynamic)
    last_err: Optional[Exception] = None
    for kwargs in candidates:
        clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        try:
            logger.info(f"Compiling model with torch.compile (device={device}, backend={compile_backend}, {clean_kwargs})...")
            compiled = torch.compile(model, backend=compile_backend, **clean_kwargs)  # type: ignore[arg-type]
            return compiled
        except Exception as e:
            last_err = e
            logger.warning(f"torch.compile {clean_kwargs} failed: {e}")

    logger.warning(f"All torch.compile attempts failed ({last_err}), using eager mode")
    return model


class HayaiOcr:
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        force_cpu: bool = False,
        quantize: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_v1: bool = False,
        use_v2: bool = False,
        backend: str = "torch",
        litert_quant: Optional[str] = None,
        litert_model_path: Optional[str] = None,
        litert_repo: Optional[str] = None,
        litert_threads: Optional[int] = None,
        compile: bool = True,
        compile_mode: Optional[str] = None,
        compile_backend: str = "inductor",
        compile_fullgraph: bool = False,
        compile_dynamic: Optional[bool] = None,
        dtype: Optional[str] = None,
        max_num_patches: Optional[int] = None,
    ):
        """
        :param pretrained_model_name_or_path: HF repo or local path for torch backend.
            Defaults to ``JustANormalTinkerer/hayai-ocr-v2.5-nova``.
            For litert backend, if provided and points to a local directory containing
            tflite files, it acts as ``litert_model_path``.
        :param force_cpu: force CPU even if GPU available (torch backend).
        :param quantize: torch weight-only quant: "int4" / "int8".
        :param device: torch device override.
        :param use_v1: use legacy v1 model (JustANormalTinkerer/hayai-ocr).
        :param use_v2: use previous v2 model (JustANormalTinkerer/hayai-ocr-v2)
            instead of the default v2.5-nova checkpoint.
        :param backend: inference backend — "torch" (default) or "litert" / "tflite".
        :param litert_quant: LiteRT quantization preset: "none"/"wi4"/"wi8_afp32"/
            "dynamic_wi4"/"dynamic_wi8" plus aliases "int4","int8","float" etc.
            Ignored for torch backend.
        :param litert_model_path: local path to LiteRT quant folder (or litert_exports root).
            If None, auto-downloads from HF.
        :param litert_repo: HF repo id hosting litert_exports (default JustANormalTinkerer/hayai-ocr-v2-tflite).
        :param litert_threads: num threads for LiteRT interpreters.
        :param compile: whether to use torch.compile for the pytorch backend (default True).
            Disable with compile=False or HAYAI_DISABLE_COMPILE=1.
        :param compile_mode: torch.compile mode override. If None, picks device-aware default:
            CPU -> "default", CUDA -> "max-autotune-no-cudagraphs" (fastest on GPU),
            with fallbacks to "reduce-overhead" / "default".
        :param compile_backend: torch.compile backend (default "inductor").
        :param compile_fullgraph: whether to require fullgraph capture (default False).
        :param compile_dynamic: whether to enable dynamic shapes (None = auto, True/False = force).
        :param dtype: inference dtype hint for autocast — "bfloat16", "float16", "float32" or None (auto).
            On CPU, "bfloat16" enables torch.cpu.amp.autocast for ~1.5-2× speedup on modern CPUs (AMX/BF16).
            On CUDA, "float16" (default) is used internally by the model. Ignored for litert backend.
        :param max_num_patches: NaFlex patch budget for the torch v2 backend
            (default 512 for v2.5-nova, 256 otherwise; override with
            ``HAYAI_MAX_NUM_PATCHES`` when None). Larger values keep more detail
            on big crops at higher compute cost; smaller values are faster.
            Ignored for v1; unsupported for litert (TFLite graphs are fixed at 256).
        """
        backend_norm = backend.strip().lower() if isinstance(backend, str) else "torch"
        if backend_norm in ("tflite", "lite_rt", "litert"):
            backend_norm = "litert"
        if backend_norm not in ("torch", "litert"):
            raise ValueError(f"Unknown backend {backend!r}. Use 'torch' or 'litert'.")
        if use_v1 and use_v2:
            raise ValueError("use_v1 and use_v2 are mutually exclusive; pass at most one.")
        self.backend = backend_norm
        self.is_litert = self.backend == "litert"
        # store compile / dtype settings for _generate_batch autocast
        self._compile = compile
        self._compile_mode = compile_mode
        self._compile_backend = compile_backend
        self._compile_fullgraph = compile_fullgraph
        self._compile_dynamic = compile_dynamic
        self._dtype_hint = dtype.strip().lower() if isinstance(dtype, str) else None

        if self.is_litert:
            self.max_num_patches = _resolve_max_num_patches(max_num_patches, default=DEFAULT_MAX_NUM_PATCHES)
            if self.max_num_patches != 256:
                raise ValueError(
                    f"max_num_patches={self.max_num_patches} is only supported on the torch backend; "
                    "LiteRT exports have a fixed 256-patch layout."
                )
            if use_v1:
                raise ValueError("LiteRT backend only supports Hayai OCR v2 (use_v1 must be False).")
            if quantize is not None:
                logger.warning("quantize is for torch backend; did you mean litert_quant? Ignoring quantize for litert.")
            _quant = litert_quant if litert_quant is not None else "wi4"
            if normalize_litert_quant is not None:
                _quant_canon = normalize_litert_quant(_quant)
            else:
                _quant_canon = _quant
            self.litert_quant = _quant_canon
            self.litert_repo = litert_repo or _LITERT_DEFAULT_REPO
            _litert_path = litert_model_path
            if _litert_path is None and pretrained_model_name_or_path is not None:
                p = Path(pretrained_model_name_or_path)
                if p.exists():
                    _litert_path = pretrained_model_name_or_path
                    logger.info(f"Using pretrained_model_name_or_path as litert_model_path: {p}")
                elif "/" not in pretrained_model_name_or_path and pretrained_model_name_or_path in (
                    "none",
                    "wi4",
                    "wi8_afp32",
                    "dynamic_wi4",
                    "dynamic_wi8",
                    "int4",
                    "int8",
                    "float",
                ):
                    self.litert_quant = normalize_litert_quant(pretrained_model_name_or_path)  # type: ignore
                    _litert_path = None
            self.litert_model_path = _litert_path
            self.pretrained_model_name_or_path = litert_model_path or pretrained_model_name_or_path or self.litert_repo
            self.is_v1 = False
            self.is_nova = False
            self.use_v2 = use_v2
            if LitertOcrEngine is None:
                raise ImportError(
                    "LiteRT backend requires optional dependencies `ai_edge_litert`, `tokenizers`, `huggingface_hub`. "
                    "Install with `pip install hayai-ocr[litert]` or `pip install ai-edge-litert tokenizers huggingface_hub`."
                )
            logger.info(f"Loading OCR model (mode: v2 litert, quant={self.litert_quant}) from {self.pretrained_model_name_or_path}")
            self.device = "cpu"
            self.processor = None  # type: ignore
            self.tokenizer = None  # type: ignore
            self.litert_engine = LitertOcrEngine(
                quant=self.litert_quant,
                model_dir=self.litert_model_path,
                hf_repo=self.litert_repo,
                num_threads=litert_threads,
            )
            self.model = self.litert_engine  # type: ignore
            example_path = Path(__file__).parent / "assets/example.jpg"
            if example_path.is_file():
                try:
                    self(example_path)
                except Exception as e:
                    logger.warning(f"LiteRT warmup failed: {e}")
            logger.info("OCR ready (litert)")
            return
        if use_v1 and pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = V1_REPO
        elif use_v2 and pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = V2_REPO
        elif pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = DEFAULT_REPO

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.is_v1 = use_v1 or (
            pretrained_model_name_or_path == V1_REPO
            or (pretrained_model_name_or_path.endswith("hayai-ocr") and "v2" not in pretrained_model_name_or_path)
        )
        self.is_nova = not self.is_v1 and _is_nova_repo(pretrained_model_name_or_path)
        self.use_v2 = use_v2

        default_patches = NOVA_DEFAULT_MAX_NUM_PATCHES if self.is_nova else DEFAULT_MAX_NUM_PATCHES
        self.max_num_patches = _resolve_max_num_patches(max_num_patches, default=default_patches)

        if self.is_v1:
            mode = "v1 (legacy)"
        elif self.is_nova:
            mode = "v2.5-nova"
        else:
            mode = "v2"
        logger.info(f"Loading OCR model (mode: {mode}) from {pretrained_model_name_or_path}")
        if self.is_v1 and self.max_num_patches != 256:
            logger.warning("max_num_patches is ignored for the v1 model (torch v2 NaFlex only).")

        if force_cpu:
            self.device = "cpu"
        elif device is not None:
            self.device = str(device)
        elif torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        if self.is_v1:
            self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            self.model = MangaVisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)
            self.model.to(self.device)
            self.model.eval()
        else:
            self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path)

            logger.info(f"Loading model on {self.device}...")
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True
            )
            self.model.to(self.device)
            apply_torchao_quantization(self.model, quantize)
            self.model.eval()

        # ----- torch.compile (device-aware, much faster on CPU & GPU) -----
        if self._compile:
            self.model = _apply_torch_compile(
                self.model,
                self.device,
                compile_mode=self._compile_mode,
                compile_fullgraph=self._compile_fullgraph,
                compile_dynamic=self._compile_dynamic,
                compile_backend=self._compile_backend,
            )
        else:
            _configure_torch_for_inference(self.device)
            logger.info("torch.compile disabled (compile=False)")

        example_path = Path(__file__).parent / "assets/example.jpg"
        if example_path.is_file():
            try:
                self(example_path)
            except Exception as e:
                logger.warning(f"Warmup failed: {e}")

        logger.info("OCR ready")

    def __call__(
        self,
        img_or_path: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.00,
        max_num_patches: Optional[int] = None,
    ) -> Union[str, List[str]]:
        if isinstance(img_or_path, (list, tuple)):
            images = []
            for item in img_or_path:
                if isinstance(item, (str, Path)):
                    img = Image.open(item).convert("RGB")
                elif isinstance(item, Image.Image):
                    img = item.convert("RGB")
                else:
                    raise ValueError(f"Each item must be a path or PIL.Image, got: {type(item)}")
                images.append(img)

            return self._generate_batch(
                images,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                max_num_patches=max_num_patches,
            )
        else:
            if isinstance(img_or_path, (str, Path)):
                img = Image.open(img_or_path).convert("RGB")
            elif isinstance(img_or_path, Image.Image):
                img = img_or_path.convert("RGB")
            else:
                raise ValueError(f"img_or_path must be a path or PIL.Image, instead got: {type(img_or_path)}")

            return self._generate_batch(
                [img],
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                max_num_patches=max_num_patches,
            )[0]

    def _generate_batch(
        self,
        pil_images: List[Image.Image],
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.00,
        max_num_patches: Optional[int] = None,
    ) -> List[str]:
        if getattr(self, "is_litert", False):
            if max_num_patches is not None and _resolve_max_num_patches(max_num_patches) != 256:
                raise ValueError(
                    "max_num_patches is only supported on the torch backend; "
                    "LiteRT exports have a fixed 256-patch layout."
                )
            mt = min(max_new_tokens, 64) if max_new_tokens > 64 else max_new_tokens
            return self.litert_engine.generate_batch(pil_images, max_new_tokens=mt, repetition_penalty=repetition_penalty)
        if self.is_v1:
            # Try batched path for v1 (faster on GPU & compiled inductor)
            if len(pil_images) > 1:
                try:
                    # AutoImageProcessor supports batched images
                    inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
                    # move with non_blocking for GPU
                    to_kwargs = {"non_blocking": True} if "cuda" in str(self.device) else {}
                    pixel_values = inputs["pixel_values"].to(self.device, **to_kwargs)  # type: ignore
                    # v1 processor returns pixel_attention_mask / spatial_shapes for NaFlex models;
                    # for pure ViT they may be absent
                    pixel_attention_mask = inputs.get("pixel_attention_mask")
                    spatial_shapes = inputs.get("spatial_shapes")
                    gen_kwargs = {
                        "pixel_values": pixel_values,
                        "max_length": 256,
                        "num_beams": 4,
                    }
                    if pixel_attention_mask is not None:
                        gen_kwargs["pixel_attention_mask"] = pixel_attention_mask.to(self.device, **to_kwargs)
                    if spatial_shapes is not None:
                        gen_kwargs["spatial_shapes"] = spatial_shapes.to(self.device, **to_kwargs)
                    with torch.inference_mode():
                        # Use autocast for faster gemm (BF16 on CPU, FP16 on CUDA)
                        if self.device == "cpu" and self._dtype_hint != "float32":
                            # Prefer bfloat16 on CPU for ~1.5-2× (requires AMX/BF16 capable CPU, falls back gracefully)
                            try:
                                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                                    out_ids = self.model.generate(**gen_kwargs)
                            except Exception:
                                out_ids = self.model.generate(**gen_kwargs)
                        elif "cuda" in str(self.device):
                            try:
                                with torch.autocast(device_type="cuda", dtype=torch.float16):
                                    out_ids = self.model.generate(**gen_kwargs)
                            except Exception:
                                out_ids = self.model.generate(**gen_kwargs)
                        else:
                            out_ids = self.model.generate(**gen_kwargs)
                    # out_ids: (batch, seq_len)
                    if isinstance(out_ids, torch.Tensor):
                        out_ids = out_ids.cpu()
                    else:
                        # HF generate may return list?
                        out_ids = torch.stack([o.cpu() if isinstance(o, torch.Tensor) else torch.tensor(o) for o in out_ids])
                    results = []
                    for seq in out_ids:
                        # tokenizer expects 1D
                        if isinstance(seq, torch.Tensor):
                            seq = seq.tolist()
                        text = self.tokenizer.decode(seq, skip_special_tokens=True)
                        results.append(post_process(text))
                    if len(results) == len(pil_images):
                        return results
                except Exception as e:
                    logger.debug(f"Batched v1 generate failed ({e}), falling back to per-image loop")

            # Fallback: per-image loop (original behaviour) but with inference_mode + autocast
            results = []
            for img in pil_images:
                inputs = self.processor(images=img, return_tensors="pt")
                to_kwargs = {"non_blocking": True} if "cuda" in str(self.device) else {}
                pv = inputs["pixel_values"].to(self.device, **to_kwargs)  # type: ignore
                pam = inputs.get("pixel_attention_mask")
                ss = inputs.get("spatial_shapes")
                gen_kwargs = {"pixel_values": pv, "max_length": 256, "num_beams": 4}
                if pam is not None:
                    gen_kwargs["pixel_attention_mask"] = pam.to(self.device, **to_kwargs)
                if ss is not None:
                    gen_kwargs["spatial_shapes"] = ss.to(self.device, **to_kwargs)
                with torch.inference_mode():
                    if self.device == "cpu" and self._dtype_hint != "float32":
                        try:
                            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                                out = self.model.generate(**gen_kwargs)
                        except Exception:
                            out = self.model.generate(**gen_kwargs)
                    elif "cuda" in str(self.device):
                        try:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                out = self.model.generate(**gen_kwargs)
                        except Exception:
                            out = self.model.generate(**gen_kwargs)
                    else:
                        out = self.model.generate(**gen_kwargs)
                    seq = out[0].cpu() if isinstance(out, torch.Tensor) else out[0].cpu() if hasattr(out[0], "cpu") else out[0]
                    if isinstance(seq, torch.Tensor):
                        seq = seq.tolist() if seq.dim() > 0 else [seq.item()]
                    text = self.tokenizer.decode(seq, skip_special_tokens=True)
                    results.append(post_process(text))
            return results
        else:
            effective_patches = (
                self.max_num_patches
                if max_num_patches is None
                else _resolve_max_num_patches(max_num_patches)
            )
            inputs = self.processor(images=pil_images, max_num_patches=effective_patches, return_tensors="pt")

            to_kwargs = {"non_blocking": True} if "cuda" in str(self.device) else {}
            pixel_values = inputs.pixel_values.to(self.device, **to_kwargs)
            pixel_attention_mask = inputs.pixel_attention_mask.to(self.device, **to_kwargs)
            spatial_shapes = inputs.spatial_shapes.to(self.device, **to_kwargs)

            # torch.inference_mode is faster than torch.no_grad (disables version counter)
            # + CPU bfloat16 autocast gives ~1.5-2× on AVX-512/BF16 / AMX CPUs.
            # HayaiModel.generate already does cuda autocast internally, but we add outer
            # cpu autocast here to accelerate the CPU path without touching HF hub code.
            with torch.inference_mode():
                if self.device == "cpu" and self._dtype_hint != "float32":
                    # bfloat16 autocast on CPU (oneDNN). Falls back gracefully on older CPUs.
                    # User can force float32 via dtype="float32" or HAYAI_DTYPE env.
                    want_bf16 = self._dtype_hint in (None, "auto", "bfloat16", "bf16")
                    # Also respect env var for easy disabling
                    if os.environ.get("HAYAI_DTYPE", "").lower() in ("float32", "fp32"):
                        want_bf16 = False
                    if want_bf16:
                        try:
                            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                                generated_texts = self.model.generate(
                                    pixel_values=pixel_values,
                                    pixel_attention_mask=pixel_attention_mask,
                                    spatial_shapes=spatial_shapes,
                                    tokenizer=self.tokenizer,
                                    max_new_tokens=max_new_tokens,
                                    repetition_penalty=repetition_penalty,
                                )
                            return generated_texts
                        except Exception as e:
                            logger.debug(f"CPU bfloat16 autocast failed ({e}), falling back to float32")
                elif "cuda" in str(self.device):
                    # For CUDA the model's internal autocast already uses float16,
                    # but wrapping in outer autocast can help inductor fuse casts.
                    # We still use inference_mode + optional outer autocast.
                    if self._dtype_hint in ("float16", "fp16", None, "auto"):
                        try:
                            with torch.autocast(device_type="cuda", dtype=torch.float16):
                                generated_texts = self.model.generate(
                                    pixel_values=pixel_values,
                                    pixel_attention_mask=pixel_attention_mask,
                                    spatial_shapes=spatial_shapes,
                                    tokenizer=self.tokenizer,
                                    max_new_tokens=max_new_tokens,
                                    repetition_penalty=repetition_penalty,
                                )
                            return generated_texts
                        except Exception as e:
                            logger.debug(f"CUDA float16 autocast failed ({e}), falling back")
                # Fallback: no outer autocast (lets model.generate handle it)
                generated_texts = self.model.generate(
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                    spatial_shapes=spatial_shapes,
                    tokenizer=self.tokenizer,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty,
                )

            return generated_texts

    def _preprocess(self, img):
        if getattr(self, "is_litert", False):
            from hayai_ocr.litert import preprocess_image_pil, compute_pos_embeds, compute_rope

            if isinstance(img, (list, tuple)):
                img = img[0]
            if isinstance(img, (str, Path)):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, Image.Image):
                img = img.convert("RGB")
            pv, mask, (ph, pw) = preprocess_image_pil(img)  # type: ignore
            pos = compute_pos_embeds(self.litert_engine.position_base, ph, pw)  # type: ignore
            cos, sin = compute_rope(ph, pw)  # type: ignore
            return {"pixel_values": pv, "attention_mask": mask, "pos_embeds": pos, "cos": cos, "sin": sin}
        if self.is_v1:
            if isinstance(img, (list, tuple)):
                img = img[0]
            inputs = self.processor(images=img, return_tensors="pt")
            return {
                "pixel_values": inputs.pixel_values,
                "pixel_attention_mask": inputs.get("pixel_attention_mask"),
                "spatial_shapes": inputs.get("spatial_shapes"),
            }
        else:
            if not isinstance(img, list):
                img = [img]
            return self.processor(images=img, max_num_patches=self.max_num_patches, return_tensors="pt")


# Backwards-compatible alias
MangaOcr = HayaiOcr


def post_process(text: str) -> str:
    """Post-processing utility for backwards compatibility."""
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = jaconv.h2z(text, ascii=True, digit=True)
    return text
