# Hayai OCR (速いOCR)

Fast optical character recognition for Japanese, Chinese, Korean, and English text, with the main focus being Japanese manga.
Powered by **Hayai OCR v2.7 Supernova** ([JustANormalTinkerer/hayai-ocr-v2.7-supernova](https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.7-supernova)), pairing Google's [SigLIP2 NaFlex](https://huggingface.co/google/siglip2-base-patch16-naflex) vision encoder with a 12-layer causal GQA decoder, multi-depth stroke fusion (`StaticDepthFusion`), dual-order `VisualCausalFlow` (horizontal LTR + vertical RTL), 226-class IDS grounding, and single-pass adaptive patch routing (`max_patches="auto"`).

Older checkpoints are still available as fallbacks (`use_v25=True` / `--use-v25` for v2.5-nova, `use_v2=True` / `--use-v2` for v2, and `use_v1=True` / `--use-v1` for legacy v1), though **v2.7 Supernova** is strongly recommended for both accuracy and speed.

> [!NOTE]
> Hayai OCR v2+ is NOT based on manga-ocr in any way. v1 is heavily inspired by manga-ocr, but is not finetuned from it.

Hayai OCR can be used as a general purpose printed Asian + English language OCR, and its main goal is high-quality crop-level text recognition, robust against scenarios specific to manga:
- both vertical and horizontal text
- text with furigana
- text overlaid on images
- wide variety of fonts and font styles
- low quality images
- SFX

Unlike many OCR models, Hayai OCR supports recognizing multi-line text in a single forward pass, so that text bubbles found in manga can be processed at once, without splitting them into lines.

> **Scope:** Hayai is designed for crop-level recognition. For full-page OCR, pair it with a text/bubble detector.

See also:
- [Poricom](https://github.com/bluaxees/Poricom), a GUI reader
- [mokuro](https://github.com/kha-white/mokuro), a tool for generating HTML overlays for manga

# Installation

You need Python 3.9 or newer. Please note that the newest Python release might not be supported due to a PyTorch
dependency, which often breaks with new Python releases and needs some time to catch up.
Refer to [PyTorch website](https://pytorch.org/get-started/locally/) for a list of supported Python versions.

If you want to run with GPU, install PyTorch as described [here](https://pytorch.org/get-started/locally/#start-locally),
otherwise this step can be skipped.

```bash
pip install hayai-ocr
```

# Usage

## Python API

```python
from hayai_ocr import HayaiOcr

mocr = HayaiOcr()
text = mocr('/path/to/img')
```

or with PIL:

```python
import PIL.Image
from hayai_ocr import HayaiOcr

mocr = HayaiOcr()
img = PIL.Image.open('/path/to/img')
text = mocr(img)
```

Batch processing (with automatic single-pass adaptive patch routing across `{128, 192, 256, 384}`):

```python
texts = mocr(['/path/to/img1.png', '/path/to/img2.png'])
```

You can also override the patch budget (`"auto"` by default on v2.7 Supernova, or fixed integers like `128`, `256`, `384`):

```python
mocr = HayaiOcr(max_patches=256)
# or per call:
text = mocr('/path/to/img', max_patches=128)
```

### Quantization (int4 / int8)

Hayai OCR supports weight-only quantization via `torchao` (PyTorch AO) to reduce VRAM usage:

- **`int8`**: INT8 weight-only — ~2x memory reduction with minimal accuracy loss
- **`int4`**: INT4 weight-only — ~4x memory reduction

```python
# Run with int4 quantization
mocr = HayaiOcr(quantize="int4")

# Run with int8 quantization
mocr = HayaiOcr(quantize="int8")
```

### ONNX Runtime backend (v2.7 Supernova, no PyTorch required)

Runs exported **v2.7 Supernova** ONNX graphs (prefill + decode with KV-cache and adaptive `max_patches="auto"` bucketing) via `onnxruntime`. Artefacts: [JustANormalTinkerer/hayai-ocr-v2.7-supernova-onnx](https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.7-supernova-onnx) (`fp32`, `fp16`, `int8`, `int4`).

```bash
pip install hayai-ocr[onnx] or pip install hayai-ocr[onnx-gpu] # installs onnxruntime, tokenizers, huggingface_hub
```

```python
from hayai_ocr import HayaiOcr

mocr = HayaiOcr(backend="onnx")  # default precision: int8
mocr = HayaiOcr(backend="onnx", onnx_precision="int4")
mocr = HayaiOcr(backend="onnx", onnx_precision="fp16")
# local exports: HayaiOcr(backend="onnx", onnx_model_path="/path/to/hayai_onnx")
```

CLI: `hayai_ocr --backend onnx --onnx-precision int8`

### LiteRT backend (legacy v2 model only)

> [!WARNING]
> The LiteRT backend currently runs the **older Hayai OCR v2 model** with a fixed 256-patch layout and does **not** include v2.7 Supernova improvements. For running **v2.7 Supernova** without PyTorch, use `backend="onnx"` above.

Alternative backend that runs exported v2 TFLite graphs (encoder + prefill/decode with KV-cache) via `ai_edge_litert`. Artefacts: [JustANormalTinkerer/hayai-ocr-v2-tflite](https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2-tflite) (`litert_exports/{none,wi4,wi8_afp32,dynamic_wi4,dynamic_wi8}`).

```bash
pip install hayai-ocr[litert]  # installs ai_edge_litert, tokenizers, huggingface_hub
```

```python
from hayai_ocr import HayaiOcr

mocr = HayaiOcr(backend="litert")  # default: wi4 (int4), runs v2 model
mocr = HayaiOcr(backend="litert", litert_quant="dynamic_wi4")
mocr = HayaiOcr(backend="litert", litert_quant="float")
# local exports: HayaiOcr(backend="litert", litert_model_path="/path/to/litert_exports/dynamic_wi4")
```

Quants: `none`/`float`, `wi4`/`int4`, `wi8_afp32`/`int8`, `dynamic_wi4`/`dynamic_int4`, `dynamic_wi8`/`dynamic_int8`. Override HF repo with `litert_repo`.

CLI: `hayai_ocr --backend litert --litert-quant dynamic_wi4`

### Older Model Fallbacks (v2.5-nova / v2 / v1)

> [!WARNING]
> `use_v25`, `use_v2`, and `use_v1` load older checkpoints with lower accuracy (and in the case of v1, Japanese-only support and much slower inference). Use them only if you specifically require backwards compatibility.

```python
# Previous v2.5-nova model (JustANormalTinkerer/hayai-ocr-v2.5-nova)
mocr = HayaiOcr(use_v25=True)

# Older v2 model (JustANormalTinkerer/hayai-ocr-v2)
mocr = HayaiOcr(use_v2=True)

# Legacy v1 model (JustANormalTinkerer/hayai-ocr)
mocr = HayaiOcr(use_v1=True)
```

> **Note:** The backwards-compatible `MangaOcr` alias is still available:
> ```python
> from hayai_ocr import MangaOcr
> mocr = MangaOcr()
> ```

## Running in the background

Hayai OCR can run in the background and process new images as they appear.

You might use a tool like [ShareX](https://getsharex.com/) or [Flameshot](https://flameshot.org/) to manually capture a region of the screen and let the
OCR read it either from the system clipboard, or a specified directory. By default, Hayai OCR will write recognized text to clipboard,
from which it can be read by a dictionary like [Yomitan](https://github.com/yomidevs/yomitan).

Clipboard mode on Linux requires `wl-copy` for Wayland sessions or `xclip` for X11 sessions. You can find out which one your system needs by running `echo $XDG_SESSION_TYPE` in the terminal.

Your full setup for reading manga with a dictionary might look like this:

capture region with ShareX -> write image to clipboard -> Hayai OCR -> write text to clipboard -> Yomitan

- To read images from clipboard and write recognized texts to clipboard, run in command line:
    ```commandline
    hayai_ocr
    ```
- To run with ONNX Runtime (v2.7 Supernova):
    ```commandline
    hayai_ocr --backend onnx --onnx-precision int8
    ```
- To run with torch quantization in CLI:
    ```commandline
    hayai_ocr --quantize int4
    ```
- To run with an older model fallback:
    ```commandline
    hayai_ocr --use-v25
    hayai_ocr --use-v2
    hayai_ocr --use-v1
    ```
- To read images from ShareX's screenshot folder, run in command line:
    ```commandline
    hayai_ocr "/path/to/sharex/screenshot/folder"
    ```

When running for the first time, downloading the model might take a few minutes.
The OCR is ready to use after `OCR ready` message appears in the logs.

- To see other options, run in command line:
    ```commandline
    hayai_ocr --help
    ```

If `hayai_ocr` doesn't work, you might also try replacing it with `python -m hayai_ocr`.

## Usage tips

- Keep `repetition_penalty=1.0` (the default): CJK text relies heavily on legitimate character repetition (e.g., onomatopoeia like ドキドキ or repeated digits). Penalties above 1.0 degrade accuracy.
- Use `max_patches="auto"` (the default for v2.7 Supernova): automatically assigns each crop in a single pass to `{128, 192, 256, 384}` patches under `max_scale=4.0`.
- Pass cropped bubbles/text boxes rather than full comic pages.
- Set `HAYAI_DISABLE_COMPILE=1` if you need to opt out of `torch.compile` on custom kernels.

## Benchmarks (v2.7 Supernova)

### JMangaBench_Read v1.3.3 (7,789 crops)

| Model | Composite ↑ | 1-NED (All) ↑ | EM % ↑ | CER % ↓ |
|---|---:|---:|---:|---:|
| **Hayai OCR v2.7 Supernova (auto)** | **98.87** | **99.23** | **95.70** | **0.96** |
| Hayai OCR v2.7 Supernova (p256) | 98.72 | 99.11 | 95.17 | 1.79 |
| Hayai OCR v2.7 Supernova (p384) | 98.69 | 99.10 | 95.03 | 1.00 |
| Hayai OCR v2.7 Supernova (p128) | 98.44 | 98.77 | 92.99 | 3.17 |
| manga-ocr-base-CPT-flex (#1 ranked) | 97.48 | — | 91.71 | 1.50 |
| Hayai v2.5 Nova (p384) | 96.63 | — | 88.20 | 2.38 |
| Hayai v2.5 Nova (p256) | 96.60 | — | 87.19 | 2.95 |

### JMangaBench_Hard v1.0 (3,015 SFX, stylized & handwritten crops)

| Model | Score ↑ | CER % ↓ | EM % ↑ |
|---|---:|---:|---:|
| **Hayai OCR v2.7 Supernova (auto)** | **96.35** | **4.02** | **89.35** |
| Hayai OCR v2.7 Supernova (p256) | 96.10 | 4.77 | 88.72 |
| manga-ocr-ar-experimental (#1 ranked) | 95.98 | 4.75 | 88.52 |
| Hayai OCR v2.7 Supernova (p128) | 95.94 | 5.03 | 88.33 |
| PaddleOCR-VL-1.6 manga LoRA v6 | 91.87 | 9.21 | 78.57 |
| Hayai v2.5 Nova (p256) | 88.87 | 11.61 | 72.34 |

### Throughput (Tesla T4, fp16 + CUDA Graphs)

- `max_patches="auto"` (JMangaBench_Read, bs=64): **103.39 images/s**, **2.27 GB** peak VRAM
- `max_patches="auto"` (JMangaBench_Hard, bs=64): **137 images/s**
- `max_patches=256` (bs=512): **147.28 images/s** (6.59 ms/crop)
- `max_patches=128` (bs=64): **208.07 images/s** (4.75 ms/crop), **0.92 GB** peak VRAM

# Acknowledgments

This project is a fork of [manga-ocr](https://github.com/kha-white/manga-ocr) by [kha-white](https://github.com/kha-white).

Training data included:
- [Manga109-s](http://www.manga109.org/en/download_s.html) dataset
- [jawildtext](https://huggingface.co/datasets/llm-jp/jawildtext) dataset
- [AnimeText](https://huggingface.co/datasets/deepghs/AnimeText) dataset
- Additional synthetic and cropped manga datasets
