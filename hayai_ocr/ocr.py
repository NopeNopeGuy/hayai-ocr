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


class HayaiOcr:
    def __init__(
        self,
        pretrained_model_name_or_path: Optional[str] = None,
        force_cpu: bool = False,
        quantize: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        use_v1: bool = False,
        backend: str = "torch",
        litert_quant: Optional[str] = None,
        litert_model_path: Optional[str] = None,
        litert_repo: Optional[str] = None,
        litert_threads: Optional[int] = None,
    ):
        """
        :param pretrained_model_name_or_path: HF repo or local path for torch backend.
            For litert backend, if provided and points to a local directory containing
            tflite files, it acts as ``litert_model_path``.
        :param force_cpu: force CPU even if GPU available (torch backend).
        :param quantize: torch weight-only quant: "int4" / "int8".
        :param device: torch device override.
        :param use_v1: use legacy v1 model.
        :param backend: inference backend — "torch" (default) or "litert" / "tflite".
        :param litert_quant: LiteRT quantization preset: "none"/"wi4"/"wi8_afp32"/
            "dynamic_wi4"/"dynamic_wi8" plus aliases "int4","int8","float" etc.
            Ignored for torch backend.
        :param litert_model_path: local path to LiteRT quant folder (or litert_exports root).
            If None, auto-downloads from HF.
        :param litert_repo: HF repo id hosting litert_exports (default JustANormalTinkerer/hayai-ocr-v2-tflite).
        :param litert_threads: num threads for LiteRT interpreters.
        """
        backend_norm = backend.strip().lower() if isinstance(backend, str) else "torch"
        if backend_norm in ("tflite", "lite_rt", "litert"):
            backend_norm = "litert"
        if backend_norm not in ("torch", "litert"):
            raise ValueError(f"Unknown backend {backend!r}. Use 'torch' or 'litert'.")
        self.backend = backend_norm
        self.is_litert = self.backend == "litert"

        if self.is_litert:
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
            pretrained_model_name_or_path = "JustANormalTinkerer/hayai-ocr"
        elif pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = "JustANormalTinkerer/hayai-ocr-v2"

        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.is_v1 = use_v1 or (
            pretrained_model_name_or_path == "JustANormalTinkerer/hayai-ocr"
            or (pretrained_model_name_or_path.endswith("hayai-ocr") and "v2" not in pretrained_model_name_or_path)
        )

        logger.info(
            f"Loading OCR model (mode: {'v1 (legacy)' if self.is_v1 else 'v2'}) from {pretrained_model_name_or_path}"
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

        self.model = torch.compile(self.model)

        example_path = Path(__file__).parent / "assets/example.jpg"
        if example_path.is_file():
            self(example_path)

        logger.info("OCR ready")

    def __call__(
        self,
        img_or_path: Union[str, Path, Image.Image, List[Union[str, Path, Image.Image]]],
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.00,
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
                images, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty
            )
        else:
            if isinstance(img_or_path, (str, Path)):
                img = Image.open(img_or_path).convert("RGB")
            elif isinstance(img_or_path, Image.Image):
                img = img_or_path.convert("RGB")
            else:
                raise ValueError(f"img_or_path must be a path or PIL.Image, instead got: {type(img_or_path)}")

            return self._generate_batch(
                [img], max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty
            )[0]

    def _generate_batch(
        self,
        pil_images: List[Image.Image],
        max_new_tokens: int = 128,
        repetition_penalty: float = 1.00,
    ) -> List[str]:
        if getattr(self, "is_litert", False):
            mt = min(max_new_tokens, 64) if max_new_tokens > 64 else max_new_tokens
            return self.litert_engine.generate_batch(pil_images, max_new_tokens=mt, repetition_penalty=repetition_penalty)
        if self.is_v1:
            results = []
            for img in pil_images:
                inputs = self.processor(img, return_tensors="pt")
                x = self.model.generate(
                    pixel_values=inputs["pixel_values"].to(self.device),
                    pixel_attention_mask=inputs["pixel_attention_mask"].to(self.device),
                    spatial_shapes=inputs["spatial_shapes"].to(self.device),
                    max_length=256,
                    num_beams=4,
                )[0].cpu()
                x = self.tokenizer.decode(x, skip_special_tokens=True)
                x = post_process(x)
                results.append(x)
            return results
        else:
            inputs = self.processor(images=pil_images, max_num_patches=256, return_tensors="pt")

            pixel_values = inputs.pixel_values.to(self.device)
            pixel_attention_mask = inputs.pixel_attention_mask.to(self.device)
            spatial_shapes = inputs.spatial_shapes.to(self.device)

            with torch.no_grad():
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
            inputs = self.processor(img, return_tensors="pt")
            return {
                "pixel_values": inputs.pixel_values,
                "pixel_attention_mask": inputs.pixel_attention_mask,
                "spatial_shapes": inputs.spatial_shapes,
            }
        else:
            if not isinstance(img, list):
                img = [img]
            return self.processor(images=img, max_num_patches=256, return_tensors="pt")


# Backwards-compatible alias
MangaOcr = HayaiOcr


def post_process(text: str) -> str:
    """Post-processing utility for backwards compatibility."""
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = jaconv.h2z(text, ascii=True, digit=True)
    return text
