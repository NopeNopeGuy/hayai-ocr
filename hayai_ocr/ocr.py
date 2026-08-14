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
    ):
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
