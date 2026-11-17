import inspect
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

try:
    from hayai_ocr.onnx import (
        OnnxOcrEngine,
        normalize_onnx_precision,
        DEFAULT_HF_REPO as _ONNX_DEFAULT_REPO,
    )
except Exception:  # pragma: no cover - optional deps missing
    OnnxOcrEngine = None  # type: ignore
    normalize_onnx_precision = None  # type: ignore
    _ONNX_DEFAULT_REPO = "JustANormalTinkerer/hayai-ocr-v2.7-supernova-onnx"


V1_REPO = "JustANormalTinkerer/hayai-ocr"
V2_REPO = "JustANormalTinkerer/hayai-ocr-v2"
V25_NOVA_REPO = "JustANormalTinkerer/hayai-ocr-v2.5-nova"
V27_SUPERNOVA_REPO = "JustANormalTinkerer/hayai-ocr-v2.7-supernova"
DEFAULT_REPO = V27_SUPERNOVA_REPO

DEFAULT_MAX_NUM_PATCHES = 256
NOVA_DEFAULT_MAX_NUM_PATCHES = 512
SUPERNOVA_DEFAULT_MAX_PATCHES = "auto"


def _is_supernova_repo(repo: Optional[str]) -> bool:
    """Return True if ``repo`` refers to the v2.7-supernova checkpoint family."""
    if not repo:
        return False
    lowered = repo.lower()
    return repo == V27_SUPERNOVA_REPO or "v2.7" in lowered or "supernova" in lowered


def _is_nova_repo(repo: Optional[str]) -> bool:
    """Return True if ``repo`` refers to the v2.5-nova checkpoint family."""
    if not repo or _is_supernova_repo(repo):
        return False
    lowered = repo.lower()
    return repo == V25_NOVA_REPO or "v2.5-nova" in lowered or "nova" in lowered


def _is_v2_repo(repo: Optional[str]) -> bool:
    """Return True if ``repo`` refers to the legacy v2 / v2.1 checkpoint family."""
    if not repo or _is_supernova_repo(repo) or _is_nova_repo(repo):
        return False
    lowered = repo.lower()
    return repo == V2_REPO or "hayai-ocr-v2" in lowered


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

def _resolve_max_num_patches(
    value: Optional[Union[int, str]],
    *,
    default: Union[int, str] = 256,
    allow_auto: bool = False,
) -> Union[int, str]:
    """Resolve ``max_patches`` / ``max_num_patches`` from an explicit value, env var, or default.

    - Explicit ``value`` (including ``0``/negative, which raises) wins.
    - Otherwise ``HAYAI_MAX_PATCHES`` or ``HAYAI_MAX_NUM_PATCHES`` env var is honored when set.
    - Otherwise ``default`` is used.
    - ``"auto"`` is accepted only when ``allow_auto=True`` (v2.7 Supernova / ONNX).
    """
    if value is None:
        env = (
            os.environ.get("HAYAI_MAX_PATCHES", "").strip()
            or os.environ.get("HAYAI_MAX_NUM_PATCHES", "").strip()
        )
        if env:
            value = env
        else:
            return default

    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped == "auto":
            if not allow_auto:
                raise ValueError(
                    "max_patches='auto' is only supported on v2.7 Supernova (torch or onnx backends)."
                )
            return "auto"
        try:
            resolved = int(stripped)
        except ValueError:
            raise ValueError(
                f"Invalid max_num_patches={value!r}: must be a positive integer"
                + (" or 'auto'" if allow_auto else "")
            )
    else:
        try:
            resolved = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise ValueError(
                f"Invalid max_num_patches={value!r}: must be a positive integer"
                + (" or 'auto'" if allow_auto else "")
            )

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
    """Return ordered list of torch.compile kwargs to try."""
    if compile_mode is not None:
        base = {"mode": compile_mode, "fullgraph": compile_fullgraph}
        if compile_dynamic is not None:
            base["dynamic"] = compile_dynamic
        return [base]

    if "cuda" in device:
        candidates: List[dict] = [
            {"mode": "max-autotune-no-cudagraphs", "fullgraph": False},
            {"mode": "max-autotune", "fullgraph": False},
            {"mode": "reduce-overhead", "fullgraph": False},
            {"mode": "default", "fullgraph": False},
        ]
    elif device == "mps":
        candidates = []
    else:  # cpu
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
    """Apply torch.compile with device-aware defaults and safe fallbacks (for legacy v1/v2/v2.5)."""
    if device == "mps":
        logger.info("torch.compile skipped on MPS (inductor not fully supported)")
        return model

    _configure_torch_for_inference(device)

    if os.environ.get("HAYAI_DISABLE_COMPILE", "0") == "1":
        logger.info("torch.compile disabled via HAYAI_DISABLE_COMPILE=1")
        return model

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
        use_v25: bool = False,
        backend: str = "torch",
        litert_quant: Optional[str] = None,
        litert_model_path: Optional[str] = None,
        litert_repo: Optional[str] = None,
        litert_threads: Optional[int] = None,
        onnx_precision: Optional[str] = None,
        onnx_model_path: Optional[str] = None,
        onnx_repo: Optional[str] = None,
        onnx_providers: Optional[List[str]] = None,
        onnx_threads: Optional[int] = None,
        compile: bool = True,
        compile_mode: Optional[str] = None,
        compile_backend: str = "inductor",
        compile_fullgraph: bool = False,
        compile_dynamic: Optional[bool] = None,
        dtype: Optional[str] = None,
        max_num_patches: Optional[Union[int, str]] = None,
        max_patches: Optional[Union[int, str]] = None,
        batch_size: int = 64,
    ):
        """
        :param pretrained_model_name_or_path: HF repo or local path for torch backend.
            Defaults to ``JustANormalTinkerer/hayai-ocr-v2.7-supernova``.
            For litert/onnx backends, if provided and points to a local directory,
            it acts as ``litert_model_path`` / ``onnx_model_path``.
        :param force_cpu: force CPU even if GPU available (torch / onnx backends).
        :param quantize: torch weight-only quant: "int4" / "int8".
        :param device: torch device override.
        :param use_v1: use legacy v1 model (JustANormalTinkerer/hayai-ocr).
        :param use_v2: use older v2 model (JustANormalTinkerer/hayai-ocr-v2)
            instead of the default v2.7-supernova checkpoint.
        :param use_v25: use previous v2.5-nova model (JustANormalTinkerer/hayai-ocr-v2.5-nova)
            instead of the default v2.7-supernova checkpoint.
        :param backend: inference backend — "torch" (default), "onnx" (v2.7 Supernova without PyTorch),
            or "litert" / "tflite" (older v2 model).
        :param litert_quant: LiteRT quantization preset: "none"/"wi4"/"wi8_afp32"/
            "dynamic_wi4"/"dynamic_wi8" plus aliases "int4","int8","float" etc.
            Ignored for torch/onnx backends.
        :param litert_model_path: local path to LiteRT quant folder (or litert_exports root).
        :param litert_repo: HF repo id hosting litert_exports (default JustANormalTinkerer/hayai-ocr-v2-tflite).
        :param litert_threads: num threads for LiteRT interpreters.
        :param onnx_precision: ONNX precision preset: "fp32", "fp16", "int8" (default), or "int4".
        :param onnx_model_path: local path to ONNX export directory. If None, auto-downloads from HF.
        :param onnx_repo: HF repo id hosting ONNX exports (default JustANormalTinkerer/hayai-ocr-v2.7-supernova-onnx).
        :param onnx_providers: ONNX Runtime execution providers override.
        :param onnx_threads: num intra-op threads for ONNX Runtime CPU sessions.
        :param compile: whether to use torch.compile for legacy PyTorch models (default True).
            v2.7 Supernova uses its built-in _FastEngine / CUDA Graphs instead of outer torch.compile.
        :param compile_mode: torch.compile mode override for legacy v1/v2/v2.5 models.
        :param compile_backend: torch.compile backend (default "inductor").
        :param compile_fullgraph: whether to require fullgraph capture (default False).
        :param compile_dynamic: whether to enable dynamic shapes (None = auto, True/False = force).
        :param dtype: inference dtype hint for autocast — "bfloat16", "float16", "float32" or None (auto).
        :param max_num_patches: alias for ``max_patches`` (kept for backwards compatibility).
        :param max_patches: NaFlex patch budget. Defaults to ``"auto"`` for v2.7 Supernova / ONNX
            (routes per crop across {128, 192, 256, 384}), ``512`` for v2.5-nova, and ``256`` for v2.
            Ignored for v1; fixed at 256 for litert. Also settable via ``HAYAI_MAX_PATCHES`` /
            ``HAYAI_MAX_NUM_PATCHES``.
        :param batch_size: default batch size for v2.7 Supernova / ONNX batched inference (default 64).
        """
        backend_norm = backend.strip().lower() if isinstance(backend, str) else "torch"
        if backend_norm in ("tflite", "lite_rt", "litert"):
            backend_norm = "litert"
        elif backend_norm in ("onnx", "ort", "onnxruntime"):
            backend_norm = "onnx"
        if backend_norm not in ("torch", "litert", "onnx"):
            raise ValueError(f"Unknown backend {backend!r}. Use 'torch', 'onnx', or 'litert'.")

        if sum(bool(x) for x in (use_v1, use_v2, use_v25)) > 1:
            raise ValueError("use_v1, use_v2, and use_v25 are mutually exclusive; pass at most one.")

        if max_patches is not None and max_num_patches is not None and max_patches != max_num_patches:
            logger.warning(
                f"Both max_patches={max_patches!r} and max_num_patches={max_num_patches!r} were passed; "
                f"using max_patches={max_patches!r}."
            )
        patch_arg = max_patches if max_patches is not None else max_num_patches

        self.backend = backend_norm
        self.is_litert = self.backend == "litert"
        self.is_onnx = self.backend == "onnx"
        self.batch_size = int(batch_size)
        self._compile = compile
        self._compile_mode = compile_mode
        self._compile_backend = compile_backend
        self._compile_fullgraph = compile_fullgraph
        self._compile_dynamic = compile_dynamic
        self._dtype_hint = dtype.strip().lower() if isinstance(dtype, str) else None

        # ----- ONNX backend (v2.7 Supernova) -----
        if self.is_onnx:
            if use_v1 or use_v2 or use_v25:
                raise ValueError(
                    "ONNX backend only supports Hayai OCR v2.7 Supernova "
                    "(use_v1, use_v2, and use_v25 must be False)."
                )
            if quantize is not None:
                logger.warning(
                    "quantize is for torch backend; did you mean onnx_precision? Ignoring quantize for onnx."
                )
            self.max_num_patches = _resolve_max_num_patches(
                patch_arg, default=SUPERNOVA_DEFAULT_MAX_PATCHES, allow_auto=True
            )
            _prec = onnx_precision if onnx_precision is not None else "int8"
            _prec_canon = normalize_onnx_precision(_prec) if normalize_onnx_precision is not None else _prec
            self.onnx_precision = _prec_canon
            self.onnx_repo = onnx_repo or _ONNX_DEFAULT_REPO
            _onnx_path = onnx_model_path
            if _onnx_path is None and pretrained_model_name_or_path is not None:
                p = Path(pretrained_model_name_or_path)
                if p.exists():
                    _onnx_path = pretrained_model_name_or_path
                    logger.info(f"Using pretrained_model_name_or_path as onnx_model_path: {p}")
                elif "/" not in pretrained_model_name_or_path and pretrained_model_name_or_path.lower() in (
                    "fp32", "float", "float32", "none", "fp16", "float16", "half", "int8", "wi8", "int4", "wi4"
                ):
                    self.onnx_precision = (
                        normalize_onnx_precision(pretrained_model_name_or_path)
                        if normalize_onnx_precision is not None
                        else pretrained_model_name_or_path.lower()
                    )
                    _onnx_path = None
            self.onnx_model_path = _onnx_path
            self.pretrained_model_name_or_path = onnx_model_path or pretrained_model_name_or_path or self.onnx_repo
            self.is_v1 = False
            self.is_v2 = False
            self.is_nova = False
            self.is_supernova = True
            self.use_v2 = False
            self.use_v25 = False

            if OnnxOcrEngine is None:
                raise ImportError(
                    "ONNX backend requires optional dependencies `onnxruntime`, `tokenizers`, `huggingface_hub`. "
                    "Install with `pip install hayai-ocr[onnx]` or `pip install onnxruntime tokenizers huggingface_hub`."
                )

            providers = onnx_providers
            if providers is None and (force_cpu or (device is not None and str(device) == "cpu")):
                providers = ["CPUExecutionProvider"]

            logger.info(
                f"Loading OCR model (mode: v2.7-supernova onnx, precision={self.onnx_precision}) "
                f"from {self.pretrained_model_name_or_path}"
            )
            self.device = "cpu" if (force_cpu or providers == ["CPUExecutionProvider"]) else "auto"
            self.processor = None  # type: ignore
            self.tokenizer = None  # type: ignore
            self.onnx_engine = OnnxOcrEngine(
                model_dir=self.onnx_model_path,
                precision=self.onnx_precision,
                hf_repo=self.onnx_repo,
                providers=providers,
                num_threads=onnx_threads,
            )
            self.model = self.onnx_engine  # type: ignore
            example_path = Path(__file__).parent / "assets/example.jpg"
            if example_path.is_file():
                try:
                    self(example_path)
                except Exception as e:
                    logger.warning(f"ONNX warmup failed: {e}")
            logger.info("OCR ready (onnx)")
            return

        # ----- LiteRT backend (legacy v2 only) -----
        if self.is_litert:
            if use_v1 or use_v25:
                raise ValueError(
                    "LiteRT backend only supports Hayai OCR v2 (use_v1 and use_v25 must be False)."
                )
            self.max_num_patches = _resolve_max_num_patches(
                patch_arg, default=DEFAULT_MAX_NUM_PATCHES, allow_auto=False
            )
            if self.max_num_patches != 256:
                raise ValueError(
                    f"max_num_patches={self.max_num_patches} is only supported on the torch/onnx backends; "
                    "LiteRT exports have a fixed 256-patch layout."
                )
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
            self.is_v2 = True
            self.is_nova = False
            self.is_supernova = False
            self.use_v2 = True
            self.use_v25 = False
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

        # ----- PyTorch backend -----
        if use_v1 and pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = V1_REPO
        elif use_v2 and pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = V2_REPO
        elif use_v25 and pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = V25_NOVA_REPO
        elif pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = DEFAULT_REPO

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.use_v2 = use_v2
        self.use_v25 = use_v25

        self.is_v1 = use_v1 or (
            pretrained_model_name_or_path == V1_REPO
            or (pretrained_model_name_or_path.endswith("hayai-ocr") and "v2" not in pretrained_model_name_or_path)
        )
        self.is_nova = not self.is_v1 and (use_v25 or _is_nova_repo(pretrained_model_name_or_path))
        self.is_v2 = not self.is_v1 and not self.is_nova and (use_v2 or _is_v2_repo(pretrained_model_name_or_path))
        self.is_supernova = (
            not self.is_v1
            and not self.is_nova
            and not self.is_v2
            and _is_supernova_repo(pretrained_model_name_or_path)
        )

        # Warnings for older model versions
        if self.is_v1:
            logger.warning(
                "Hayai OCR v1 (JustANormalTinkerer/hayai-ocr) is a legacy model that is Japanese-only, "
                "slower, and significantly less accurate. Consider using the default v2.7 Supernova model."
            )
        elif self.is_v2:
            logger.warning(
                "Hayai OCR v2 (JustANormalTinkerer/hayai-ocr-v2) is an older model version. "
                "Consider using the default v2.7 Supernova model for substantially higher accuracy and speed."
            )
        elif self.is_nova:
            logger.warning(
                "Hayai OCR v2.5-nova (JustANormalTinkerer/hayai-ocr-v2.5-nova) is an older model version. "
                "Consider using the default v2.7 Supernova model for higher accuracy, adaptive patch routing, "
                "and stronger CJK + English recognition."
            )

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
            self.max_num_patches = _resolve_max_num_patches(
                patch_arg, default=DEFAULT_MAX_NUM_PATCHES, allow_auto=False
            )
            if self.max_num_patches != 256:
                logger.warning("max_num_patches is ignored for the v1 model (torch v2+ NaFlex only).")
            logger.info(f"Loading OCR model (mode: v1 (legacy)) from {pretrained_model_name_or_path}")
            self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
            self.model = MangaVisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)
            self.model.to(self.device)
            self.model.eval()
        elif self.is_nova or self.is_v2:
            default_patches = NOVA_DEFAULT_MAX_NUM_PATCHES if self.is_nova else DEFAULT_MAX_NUM_PATCHES
            self.max_num_patches = _resolve_max_num_patches(
                patch_arg, default=default_patches, allow_auto=False
            )
            mode = "v2.5-nova" if self.is_nova else "v2"
            logger.info(f"Loading OCR model (mode: {mode}) from {pretrained_model_name_or_path}")
            self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path)

            logger.info(f"Loading model on {self.device}...")
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True
            )
            self.model.to(self.device)
            apply_torchao_quantization(self.model, quantize)
            self.model.eval()
        else:
            # v2.7 Supernova (default) or custom checkpoint
            logger.info(
                f"Loading OCR model (mode: {'v2.7-supernova' if self.is_supernova else 'auto'}) "
                f"from {pretrained_model_name_or_path}"
            )
            if self.is_supernova:
                self.processor = AutoProcessor.from_pretrained(
                    pretrained_model_name_or_path, trust_remote_code=True
                )
                self.tokenizer = getattr(self.processor, "tokenizer", None)
                if self.tokenizer is None:
                    try:
                        self.tokenizer = AutoTokenizer.from_pretrained(
                            pretrained_model_name_or_path, trust_remote_code=True
                        )
                    except Exception:
                        self.tokenizer = None
            else:
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        pretrained_model_name_or_path, trust_remote_code=True
                    )
                    self.tokenizer = getattr(self.processor, "tokenizer", None)
                except Exception:
                    self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-naflex")
                    self.tokenizer = PreTrainedTokenizerFast.from_pretrained(pretrained_model_name_or_path)

            logger.info(f"Loading model on {self.device}...")
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=True
            )
            if hasattr(self.model, "ocr"):
                self.is_supernova = True
            if self.is_supernova and hasattr(self.model, "drop_ids_head"):
                try:
                    self.model.drop_ids_head()
                except Exception as e:
                    logger.debug(f"drop_ids_head() skipped: {e}")

            self.model.to(self.device)
            apply_torchao_quantization(self.model, quantize)
            self.model.eval()

            if self.is_supernova:
                self.max_num_patches = _resolve_max_num_patches(
                    patch_arg, default=SUPERNOVA_DEFAULT_MAX_PATCHES, allow_auto=True
                )
            else:
                self.max_num_patches = _resolve_max_num_patches(
                    patch_arg, default=DEFAULT_MAX_NUM_PATCHES, allow_auto=False
                )

        # ----- torch.compile -----
        # v2.7 Supernova uses its built-in _FastEngine / CUDA Graphs and internal kernel compile;
        # wrapping it in outer torch.compile would interfere with vision_encoder inspection & CUDA graphs.
        if self.is_supernova:
            _configure_torch_for_inference(self.device)
            logger.debug("Skipping outer torch.compile for v2.7 Supernova (uses built-in _FastEngine)")
        elif self._compile:
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
        max_num_patches: Optional[Union[int, str]] = None,
        max_patches: Optional[Union[int, str]] = None,
    ) -> Union[str, List[str]]:
        if max_patches is not None and max_num_patches is not None and max_patches != max_num_patches:
            logger.warning(
                f"Both max_patches={max_patches!r} and max_num_patches={max_num_patches!r} were passed; "
                f"using max_patches={max_patches!r}."
            )
        patch_arg = max_patches if max_patches is not None else max_num_patches

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
                max_num_patches=patch_arg,
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
                max_num_patches=patch_arg,
            )[0]

    def _run_supernova_ocr(
        self,
        pil_images: List[Image.Image],
        effective_patches: Union[int, str],
        max_new_tokens: int,
        repetition_penalty: float,
    ) -> List[str]:
        use_cuda_graph = "cuda" in str(self.device)
        ocr_kwargs: dict = {
            "max_patches": effective_patches,
            "max_new_tokens": max_new_tokens,
            "use_cuda_graph": use_cuda_graph,
        }
        if len(pil_images) > 1:
            ocr_kwargs["batch_size"] = self.batch_size
            ocr_kwargs["escalate"] = False

        if repetition_penalty != 1.0:
            logger.warning(
                f"repetition_penalty={repetition_penalty} is discouraged on v2.7 Supernova "
                "(CJK text relies on legitimate character repetition; keep repetition_penalty=1.0)."
            )
            try:
                sig = inspect.signature(self.model.ocr)
                if "repetition_penalty" in sig.parameters:
                    ocr_kwargs["repetition_penalty"] = repetition_penalty
            except Exception:
                pass

        inp = pil_images[0] if len(pil_images) == 1 else pil_images

        def _invoke():
            res = self.model.ocr(inp, self.processor, **ocr_kwargs)
            return [res] if isinstance(res, str) else list(res)

        with torch.inference_mode():
            if self.device == "cpu" and self._dtype_hint != "float32":
                want_bf16 = self._dtype_hint in (None, "auto", "bfloat16", "bf16")
                if os.environ.get("HAYAI_DTYPE", "").lower() in ("float32", "fp32"):
                    want_bf16 = False
                if want_bf16:
                    try:
                        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                            return _invoke()
                    except Exception as e:
                        logger.debug(f"CPU bfloat16 autocast failed ({e}), falling back to float32")
            elif "cuda" in str(self.device) and self._dtype_hint != "float32":
                cuda_dtype = torch.bfloat16 if self._dtype_hint in ("bfloat16", "bf16") else torch.float16
                try:
                    with torch.autocast(device_type="cuda", dtype=cuda_dtype):
                        return _invoke()
                except Exception as e:
                    logger.debug(f"CUDA autocast failed ({e}), falling back")

            return _invoke()

    def _generate_batch(
        self,
        pil_images: List[Image.Image],
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.00,
        max_num_patches: Optional[Union[int, str]] = None,
    ) -> List[str]:
        if getattr(self, "is_onnx", False):
            effective_patches = (
                self.max_num_patches
                if max_num_patches is None
                else _resolve_max_num_patches(max_num_patches, default=SUPERNOVA_DEFAULT_MAX_PATCHES, allow_auto=True)
            )
            if repetition_penalty != 1.0:
                logger.warning(
                    "repetition_penalty != 1.0 is not supported by the ONNX backend "
                    "(and is discouraged for v2.7 Supernova); using greedy decoding (1.0)."
                )
            res = self.onnx_engine.ocr(
                pil_images,
                max_patches=effective_patches,
                max_new_tokens=max_new_tokens,
                batch_size=self.batch_size,
            )
            return [res] if isinstance(res, str) else list(res)

        if getattr(self, "is_litert", False):
            if max_num_patches is not None and _resolve_max_num_patches(max_num_patches, allow_auto=False) != 256:
                raise ValueError(
                    "max_num_patches is only supported on the torch/onnx backends; "
                    "LiteRT exports have a fixed 256-patch layout."
                )
            mt = min(max_new_tokens, 64) if max_new_tokens > 64 else max_new_tokens
            return self.litert_engine.generate_batch(pil_images, max_new_tokens=mt, repetition_penalty=repetition_penalty)

        if self.is_v1:
            # Try batched path for v1 (faster on GPU & compiled inductor)
            if len(pil_images) > 1:
                try:
                    inputs = self.processor(images=pil_images, return_tensors="pt", padding=True)
                    to_kwargs = {"non_blocking": True} if "cuda" in str(self.device) else {}
                    pixel_values = inputs["pixel_values"].to(self.device, **to_kwargs)  # type: ignore
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
                        if self.device == "cpu" and self._dtype_hint != "float32":
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
                    if isinstance(out_ids, torch.Tensor):
                        out_ids = out_ids.cpu()
                    else:
                        out_ids = torch.stack([o.cpu() if isinstance(o, torch.Tensor) else torch.tensor(o) for o in out_ids])
                    results = []
                    for seq in out_ids:
                        if isinstance(seq, torch.Tensor):
                            seq = seq.tolist()
                        text = self.tokenizer.decode(seq, skip_special_tokens=True)
                        results.append(post_process(text))
                    if len(results) == len(pil_images):
                        return results
                except Exception as e:
                    logger.debug(f"Batched v1 generate failed ({e}), falling back to per-image loop")

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

        if self.is_supernova:
            effective_patches = (
                self.max_num_patches
                if max_num_patches is None
                else _resolve_max_num_patches(
                    max_num_patches, default=SUPERNOVA_DEFAULT_MAX_PATCHES, allow_auto=True
                )
            )
            return self._run_supernova_ocr(
                pil_images,
                effective_patches=effective_patches,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
            )

        # Legacy v2 / v2.5-nova path
        effective_patches = (
            self.max_num_patches
            if max_num_patches is None
            else _resolve_max_num_patches(max_num_patches, allow_auto=False)
        )
        inputs = self.processor(images=pil_images, max_num_patches=effective_patches, return_tensors="pt")

        to_kwargs = {"non_blocking": True} if "cuda" in str(self.device) else {}
        pixel_values = inputs.pixel_values.to(self.device, **to_kwargs)
        pixel_attention_mask = inputs.pixel_attention_mask.to(self.device, **to_kwargs)
        spatial_shapes = inputs.spatial_shapes.to(self.device, **to_kwargs)

        with torch.inference_mode():
            if self.device == "cpu" and self._dtype_hint != "float32":
                want_bf16 = self._dtype_hint in (None, "auto", "bfloat16", "bf16")
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
        if getattr(self, "is_onnx", False):
            if not isinstance(img, (list, tuple)):
                img = [img]
            pil_list = [
                Image.open(x).convert("RGB") if isinstance(x, (str, Path)) else x.convert("RGB")
                for x in img
            ]
            pv, pos_embeds, spatial_shapes, l_vis_arange = self.onnx_engine._preprocess(
                pil_list, max_patches=self.max_num_patches
            )
            return {
                "pixel_values": pv,
                "pos_embeds": pos_embeds,
                "spatial_shapes": spatial_shapes,
                "l_vis_arange": l_vis_arange,
            }
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
        if self.is_supernova:
            if not isinstance(img, list):
                img = [img]
            try:
                return self.processor(images=img, max_patches=self.max_num_patches, return_tensors="pt")
            except TypeError:
                return self.processor(images=img, return_tensors="pt")
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
