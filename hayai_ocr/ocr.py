import re
from pathlib import Path

import jaconv
import torch
from PIL import Image
from loguru import logger
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, GenerationMixin


class MangaVisionEncoderDecoderModel(VisionEncoderDecoderModel, GenerationMixin):
    """Custom VisionEncoderDecoderModel that forwards SigLIP2 NaFlex-specific
    tensors (pixel_attention_mask, spatial_shapes) to the encoder."""

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


class HayaiOcr:
    def __init__(self, pretrained_model_name_or_path="JustANormalTinkerer/hayai-ocr", force_cpu=False):
        logger.info(f"Loading OCR model from {pretrained_model_name_or_path}")
        self.processor = AutoImageProcessor.from_pretrained(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = MangaVisionEncoderDecoderModel.from_pretrained(pretrained_model_name_or_path)

        if not force_cpu and torch.cuda.is_available():
            logger.info("Using CUDA")
            self.model.cuda()
        elif not force_cpu and torch.backends.mps.is_available():
            logger.info("Using MPS")
            self.model.to("mps")
        else:
            logger.info("Using CPU")

        example_path = Path(__file__).parent / "assets/example.jpg"
        if not example_path.is_file():
            raise FileNotFoundError(f"Missing example image {example_path}")
        self(example_path)

        logger.info("OCR ready")

    def __call__(self, img_or_path):
        if isinstance(img_or_path, (str, Path)):
            img = Image.open(img_or_path)
        elif isinstance(img_or_path, Image.Image):
            img = img_or_path
        else:
            raise ValueError(f"img_or_path must be a path or PIL.Image, instead got: {img_or_path}")

        img = img.convert("RGB")

        inputs = self._preprocess(img)
        x = self.model.generate(
            pixel_values=inputs["pixel_values"].to(self.model.device),
            pixel_attention_mask=inputs["pixel_attention_mask"].to(self.model.device),
            spatial_shapes=inputs["spatial_shapes"].to(self.model.device),
            max_length=256,
            num_beams=4,
        )[0].cpu()
        x = self.tokenizer.decode(x, skip_special_tokens=True)
        x = post_process(x)
        return x

    def _preprocess(self, img):
        inputs = self.processor(img, return_tensors="pt")
        return {
            "pixel_values": inputs.pixel_values,
            "pixel_attention_mask": inputs.pixel_attention_mask,
            "spatial_shapes": inputs.spatial_shapes,
        }


# Backwards-compatible alias
MangaOcr = HayaiOcr


def post_process(text):
    text = "".join(text.split())
    text = text.replace("…", "...")
    text = re.sub("[・.]{2,}", lambda x: (x.end() - x.start()) * ".", text)
    text = jaconv.h2z(text, ascii=True, digit=True)

    return text
