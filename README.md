# Hayai OCR (速いOCR)

Fast optical character recognition for Japanese text, with the main focus being Japanese manga.
A fork of [manga-ocr](https://github.com/kha-white/manga-ocr) by kha-white, rebuilt with a
SigLIP2 + BERT encoder-decoder architecture for improved accuracy and speed.

It uses a custom end-to-end model built with Transformers' [Vision Encoder Decoder](https://huggingface.co/docs/transformers/model_doc/vision-encoder-decoder) framework,
pairing a [SigLIP2 NaFlex](https://huggingface.co/google/siglip2-base-patch16-naflex) vision encoder with a
[Japanese BERT](https://huggingface.co/tohoku-nlp/bert-base-japanese-char-v3) character-level decoder.

Hayai OCR can be used as a general purpose printed Japanese OCR, but its main goal is to provide high quality
text recognition, robust against various scenarios specific to manga:
- both vertical and horizontal text
- text with furigana
- text overlaid on images
- wide variety of fonts and font styles
- low quality images

Unlike many OCR models, Hayai OCR supports recognizing multi-line text in a single forward pass,
so that text bubbles found in manga can be processed at once, without splitting them into lines.

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

or

```python
import PIL.Image
from hayai_ocr import HayaiOcr

mocr = HayaiOcr()
img = PIL.Image.open('/path/to/img')
text = mocr(img)
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

Your full setup for reading manga in Japanese with a dictionary might look like this:

capture region with ShareX -> write image to clipboard -> Hayai OCR -> write text to clipboard -> Yomitan

- To read images from clipboard and write recognized texts to clipboard, run in command line:
    ```commandline
    hayai_ocr
    ```
- To read images from ShareX's screenshot folder, run in command line:
    ```commandline
    hayai_ocr "/path/to/sharex/screenshot/folder"
    ```
Note that when running in the clipboard scanning mode, any image that you copy to clipboard will be processed by OCR and replaced
by recognized text. If you want to be able to copy and paste images as usual, you should use the folder scanning mode instead
and define a separate task in ShareX just for OCR, which saves screenshots to some folder without copying them to clipboard.

When running for the first time, downloading the model might take a few minutes.
The OCR is ready to use after `OCR ready` message appears in the logs.

- To see other options, run in command line:
    ```commandline
    hayai_ocr --help
    ```

If `hayai_ocr` doesn't work, you might also try replacing it with `python -m hayai_ocr`.

## Usage tips

- OCR supports multi-line text, but the longer the text, the more likely some errors are to occur.
  If the recognition failed for some part of a longer text, you might try to run it on a smaller portion of the image.
- The model was trained specifically to handle manga, visual novel, general anime and handwritten Japanese texts. It should perform well everywhere.
- The model always attempts to recognize some text on the image, even if there is none.
  Because it uses a transformer decoder (and therefore has some understanding of the Japanese language),
  it might even "dream up" some realistically looking sentences! This shouldn't be a problem for most use cases.

# Examples

Here are some cherry-picked examples showing the capability of the model.

| image                | Result |
|----------------------|--------|
| ![](assets/examples/00.jpg) | 素直にあやまるしか |
| ![](assets/examples/01.jpg) | 立川で見た〝穴〟の下の巨大な眼は： |
| ![](assets/examples/02.jpg) | 実戦剣術も一流です |
| ![](assets/examples/03.jpg) | 第３０話重苦しい闇の奥で静かに呼吸づきながら |
| ![](assets/examples/04.jpg) | よかったじゃないわよ！何逃げてるのよ！！早くあいつを退治してよ！ |
| ![](assets/examples/05.jpg) | ぎゃっ |
| ![](assets/examples/06.jpg) | ピンポーーン |
| ![](assets/examples/07.jpg) | ＬＩＮＫ！私達７人の力でガノンの塔の結界をやぶります |
| ![](assets/examples/08.jpg) | ファイアパンチ |
| ![](assets/examples/09.jpg) | 少し黙っている |
| ![](assets/examples/10.jpg) | わかるかな〜？ |
| ![](assets/examples/11.jpg) | 警察にも先生にも町中の人達に！！ |

# Acknowledgments

This project is a fork of [manga-ocr](https://github.com/kha-white/manga-ocr) by [kha-white](https://github.com/kha-white).

Training data included:
- [Manga109-s](http://www.manga109.org/en/download_s.html) dataset
- [CC-100](https://data.statmt.org/cc-100/) dataset
- [jawildtext](https://huggingface.co/datasets/llm-jp/jawildtext) dataset
- Additional synthetic and cropped manga datasets
