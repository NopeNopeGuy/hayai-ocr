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
- SFX

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
- Normalize output for english. 

# Examples

Here are some examples showing the capability of the model. 

Note: All the example images are picked randomly from Youtube videos and raw manga sites. The model has never seen these images before.
Some images (especially the youtube ones) weren't even in the scope of this project, but the model is just that good at it.  

| image                | hayai-ocr | PaddleOCR-VL For Manga |
|----------------------|--------| ----------- | 
| ![](assets/examples/01.png) | 知らない世界で見つけたメージを | 知らない世界で見つけた\n イメージを |
| ![](assets/examples/02.png) | カナデトモスソラ（Ｋａｎａｄｅｔｏｍｏｓｕｓｏｒａ） | カナデトモスリラ(Kanadetomosusora) |
| ![](assets/examples/03.png) | 建設会社社員行方 | 建設会社社員行 |
| ![](assets/examples/04.png) | だとしてもこのレベルがウロつくなんて．．．おそらく２級の呪い | だとしてもこのレベルがウロつくなんて･･･おそらく２級の呪い |
| ![](assets/examples/05.png) | パチパチパチパチ | アデアデデアデデアデ |
| ![](assets/examples/06.png) | バビュン | 川ビュン |
| ![](assets/examples/07.png) | 僕の過去とか未来とか | 僕の過去とか未来とか |
| ![](assets/examples/08.png) | くらべられっ子 | くらべられっ子 |
| ![](assets/examples/09.png) | そうだクラス分けがあるんだった！！ | そうだクラス分けがあるんだった！！ |
| ![](assets/examples/10.png) | 脇役よ、主役を超えよ！ | 脇役よ、主役を超えよ! |
| ![](assets/examples/11.png) | Ｅｈ～Ｉｄｏｎ＇ｔｒｅａｌｌｙｗａｎｔｔｏ～ | Ｅｈ～Ｉ don't really want to～ |
| ![](assets/examples/12.png) |「Ｓｏｒｒｙｆｏｒｔｈｅｗａｉｔ！Ｄｉｄｙｏｕｗａｉｔｌｏｎｇ？」| 「Sorry for the wait~!Did you wait long?」 |
| ![](assets/examples/13.png) | ＹａｍａｔｅＡｒｅａＮｅｗｒｅｓｉｄｅｎｔｉａｌｄｉｓｔｒｉｃｔｆｏｒｆｏｒｅｉｇｎｅｒｓ| Yamate Area New residential district for foreigners |

## Goals

While PaddleOCR-VL is also very accurate, it is 9x the size of this model and it struggles with SFX. The goal of this model is not to be State of the Art, but rather be usuable and fast at scale.

# Acknowledgments

This project is a fork of [manga-ocr](https://github.com/kha-white/manga-ocr) by [kha-white](https://github.com/kha-white).

Training data included:
- [Manga109-s](http://www.manga109.org/en/download_s.html) dataset
- [jawildtext](https://huggingface.co/datasets/llm-jp/jawildtext) dataset
- [AnimeText](https://huggingface.co/datasets/deepghs/AnimeText) dataset
- Additional synthetic and cropped manga datasets
