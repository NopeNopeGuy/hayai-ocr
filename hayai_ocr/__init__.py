from ._version import __version__ as __version__
from hayai_ocr.ocr import HayaiOcr as HayaiOcr

# Backwards-compatible alias
from hayai_ocr.ocr import MangaOcr as MangaOcr

try:
    from hayai_ocr.litert import LitertOcrEngine as LitertOcrEngine  # noqa: F401
except Exception:
    LitertOcrEngine = None  # type: ignore
