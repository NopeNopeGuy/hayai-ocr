import sys
import time
from pathlib import Path

import fire
import numpy as np
import pyperclip
from PIL import Image
from PIL import UnidentifiedImageError
from loguru import logger

from hayai_ocr import HayaiOcr


def are_images_identical(img1, img2):
    if None in (img1, img2):
        return img1 == img2

    img1 = np.array(img1)
    img2 = np.array(img2)

    return (img1.shape == img2.shape) and (img1 == img2).all()


def process_and_write_results(mocr, img_or_path, write_to):
    t0 = time.time()
    text = mocr(img_or_path)
    t1 = time.time()

    logger.info(f"Text recognized in {t1 - t0:0.03f} s: {text}")

    if write_to == "clipboard":
        pyperclip.copy(text)
    else:
        write_to = Path(write_to)
        if write_to.suffix != ".txt":
            raise ValueError('write_to must be either "clipboard" or a path to a text file')

        with write_to.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


def get_path_key(path):
    return path, path.lstat().st_mtime


def run(
    read_from="clipboard",
    write_to="clipboard",
    pretrained_model_name_or_path=None,
    force_cpu=False,
    quantize=None,
    use_v1=False,
    backend="torch",
    litert_quant=None,
    litert_model_path=None,
    litert_repo=None,
    litert_threads=None,
    delay_secs=0.1,
    verbose=False,
):
    """
    Run OCR in the background, waiting for new images to appear either in system clipboard, or a directory.
    Recognized texts can be either saved to system clipboard, or appended to a text file.

    :param read_from: Specifies where to read input images from. Can be either "clipboard", or a path to a directory.
    :param write_to: Specifies where to save recognized texts to. Can be either "clipboard", or a path to a text file.
    :param pretrained_model_name_or_path: Path to a trained model, either local or from Transformers' model hub.
    :param force_cpu: If True, OCR will use CPU even if GPU is available.
    :param quantize: Optional quantization type: "int4" or "int8".
    :param use_v1: If True, uses the legacy Hayai OCR v1 model (JustANormalTinkerer/hayai-ocr).
    :param backend: Inference backend: "torch" (default) or "litert" / "tflite".
    :param litert_quant: LiteRT quantization preset: "none", "wi4", "wi8_afp32", "dynamic_wi4", "dynamic_wi8"
        (aliases: "int4", "int8", "float" etc). Default "wi4".
    :param litert_model_path: Local path to LiteRT quant folder (or litert_exports root).
    :param litert_repo: HF repo id for LiteRT artefacts (default JustANormalTinkerer/hayai-ocr-v2-tflite).
    :param litert_threads: Num threads for LiteRT interpreters.
    :param verbose: If True, unhides all warnings.
    :param delay_secs: How often to check for new images, in seconds.
    """

    mocr = HayaiOcr(
        pretrained_model_name_or_path=pretrained_model_name_or_path,
        force_cpu=force_cpu,
        quantize=quantize,
        use_v1=use_v1,
        backend=backend,
        litert_quant=litert_quant,
        litert_model_path=litert_model_path,
        litert_repo=litert_repo,
        litert_threads=litert_threads,
    )

    if sys.platform not in ("darwin", "win32") and write_to == "clipboard":
        # Check if the system is using Wayland
        import os

        if os.environ.get("WAYLAND_DISPLAY"):
            # Check if the wl-clipboard package is installed
            if os.system("which wl-copy > /dev/null") == 0:
                pyperclip.set_clipboard("wl-clipboard")
            else:
                msg = (
                    "Your session uses wayland and does not have wl-clipboard installed. "
                    "Install wl-clipboard for write in clipboard to work."
                )
                raise NotImplementedError(msg)

    if read_from == "clipboard":
        from PIL import ImageGrab

        logger.info("Reading from clipboard")

        img = None
        while True:
            old_img = img

            try:
                img = ImageGrab.grabclipboard()
            except OSError as error:
                if not verbose and "cannot identify image file" in str(error):
                    # Pillow error when clipboard hasn't changed since last grab (Linux)
                    pass
                elif not verbose and "target image/png not available" in str(error):
                    # Pillow error when clipboard contains text (Linux, X11)
                    pass
                else:
                    logger.warning("Error while reading from clipboard ({})".format(error))
            else:
                if isinstance(img, Image.Image) and not are_images_identical(img, old_img):
                    process_and_write_results(mocr, img, write_to)

            time.sleep(delay_secs)

    else:
        read_from = Path(read_from)
        if not read_from.is_dir():
            raise ValueError('read_from must be either "clipboard" or a path to a directory')

        logger.info(f"Reading from directory {read_from}")

        old_paths = set()
        for path in read_from.iterdir():
            old_paths.add(get_path_key(path))

        while True:
            for path in read_from.iterdir():
                path_key = get_path_key(path)
                if path_key not in old_paths:
                    old_paths.add(path_key)

                    try:
                        img = Image.open(path)
                        img.load()
                    except (UnidentifiedImageError, OSError) as e:
                        logger.warning(f"Error while reading file {path}: {e}")
                    else:
                        process_and_write_results(mocr, img, write_to)

            time.sleep(delay_secs)


if __name__ == "__main__":
    fire.Fire(run)
