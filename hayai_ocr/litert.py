"""LiteRT backend for Hayai OCR v2.

Optional TFLite/LiteRT inference with KV-cache prefill + decode graphs
exported via ``export_kv.py`` in hayai-ocr-tflite.

Artefacts: https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2-tflite/tree/main/litert_exports
Subfolders: none (float), wi4 (int4), wi8_afp32 (int8), dynamic_wi4, dynamic_wi8.
Each contains encoder/prefill/decode .tflite, position_base.npy, tokenizer.json.

Usage:
    HayaiOcr(backend="litert", litert_quant="wi4")
    LitertOcrEngine(quant="wi4").generate(image)

Requires: ai_edge_litert, tokenizers, huggingface_hub.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from PIL import Image
from loguru import logger

PATCH_SIZE = 16
MAX_PATCHES = 256
MAX_TEXT = 64
TOTAL = MAX_PATCHES + MAX_TEXT  # 320
D_AXIS = 32  # d_head // 2
ROPE_THETA = 10000.0

DEFAULT_HF_REPO = "JustANormalTinkerer/hayai-ocr-v2-tflite"

CANONICAL_QUANTS = {"none", "wi4", "wi8_afp32", "dynamic_wi4", "dynamic_wi8"}
QUANT_ALIASES: Dict[str, str] = {
    "none": "none",
    "float": "none",
    "fp32": "none",
    "no": "none",
    "wi4": "wi4",
    "int4": "wi4",
    "wi4_afp32": "wi4",
    "weight_only_wi4_afp32": "wi4",
    "dynamic_wi4": "dynamic_wi4",
    "dynamic_wi4_afp32": "dynamic_wi4",
    "dynamic_int4": "dynamic_wi4",
    "wi8": "wi8_afp32",
    "int8": "wi8_afp32",
    "wi8_afp32": "wi8_afp32",
    "weight_only_wi8_afp32": "wi8_afp32",
    "dynamic_wi8": "dynamic_wi8",
    "dynamic_wi8_afp32": "dynamic_wi8",
    "dynamic_int8": "dynamic_wi8",
}
QUANT_SUFFIX = {
    "none": "float",
    "wi4": "int4",
    "wi8_afp32": "int8",
    "dynamic_wi4": "dynamic_wi4",
    "dynamic_wi8": "dynamic_wi8",
}


def normalize_litert_quant(quant: Optional[str]) -> str:
    """Normalize user supplied quant string to canonical folder name.

    Returns one of ``CANONICAL_QUANTS``.
    Raises ValueError for unknown values.
    """
    if quant is None:
        return "wi4"
    q = quant.strip().lower()
    if q in QUANT_ALIASES:
        return QUANT_ALIASES[q]
    raise ValueError(
        f"Unknown litert_quant {quant!r}. Valid choices: {sorted(CANONICAL_QUANTS)} "
        f"plus aliases {sorted(QUANT_ALIASES)}"
    )


def quant_to_suffix(quant: str) -> str:
    canon = normalize_litert_quant(quant)
    return QUANT_SUFFIX[canon]


# Pre-processing & RoPE (numpy)
def _get_scaled_size(h: int, w: int) -> tuple[int, int]:
    EPS = 1e-5

    def scaled(s: float, sz: int) -> int:
        return max(math.ceil((sz * s) / PATCH_SIZE) * PATCH_SIZE, PATCH_SIZE)

    smin, smax = EPS / 10, 100.0
    while smax - smin >= EPS:
        s = (smin + smax) / 2
        th = scaled(s, h)
        tw = scaled(s, w)
        if (th // 16) * (tw // 16) <= MAX_PATCHES:
            smin = s
        else:
            smax = s
    return scaled(smin, h), scaled(smin, w)


def preprocess_image_pil(pil_image: Image.Image) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Preprocess a PIL image to (pixel_values, attention_mask, (ph,pw)).

    pixel_values: (1,256,768) float32, normalized (x-0.5)/0.5
    attention_mask: (1,256) int64 1=valid, 0=pad
    """
    img = pil_image.convert("RGB")
    w0, h0 = img.size
    th, tw = _get_scaled_size(h0, w0)
    ph, pw = th // PATCH_SIZE, tw // PATCH_SIZE
    n = ph * pw
    if img.size != (tw, th):
        img = img.resize((tw, th), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) * (1.0 / 255.0)
    arr = (arr - 0.5) / 0.5
    arr = arr.reshape(ph, PATCH_SIZE, pw, PATCH_SIZE, 3).transpose(0, 2, 1, 3, 4).reshape(n, 768)
    pv = np.zeros((1, MAX_PATCHES, 768), dtype=np.float32)
    pv[0, :n] = arr
    mask = np.zeros((1, MAX_PATCHES), dtype=np.int64)
    mask[0, :n] = 1
    return pv, mask, (ph, pw)


def compute_pos_embeds(base: np.ndarray, ph: int, pw: int) -> np.ndarray:
    """Compute position embeddings for given ph,pw using base grid.

    base: (256,768) or (16,16,768) float32
    returns: (1,256,768) float32
    """
    if base.ndim == 2:
        base = base.reshape(16, 16, 768)
    if ph == 16 and pw == 16:
        out = base.reshape(MAX_PATCHES, 768)
    else:
        try:
            resample = Image.Resampling.BILINEAR
        except AttributeError:
            resample = Image.BILINEAR  # type: ignore
        out = np.empty((ph, pw, 768), dtype=np.float32)
        for c in range(768):
            img = Image.fromarray(base[:, :, c], mode="F")
            out[:, :, c] = np.array(img.resize((pw, ph), resample), dtype=np.float32)
        out = out.reshape(ph * pw, 768)
    pos = np.zeros((1, MAX_PATCHES, 768), dtype=np.float32)
    n = ph * pw
    pos[0, :n] = out
    if n < MAX_PATCHES:
        pos[0, n:] = out[0:1]
    return pos


def compute_rope(ph: int, pw: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute RoPE cos/sin for vision + text.

    Returns cos, sin each shape (1,TOTAL,D_AXIS) float32.
    """
    freqs = 1.0 / (ROPE_THETA ** (np.arange(0, D_AXIS, 2, dtype=np.float32) / D_AXIS))
    gy = np.outer(np.arange(ph, dtype=np.float32), freqs)
    gx = np.outer(np.arange(pw, dtype=np.float32), freqs)
    gy = np.expand_dims(gy, 1).repeat(pw, axis=1)
    gx = np.expand_dims(gx, 0).repeat(ph, axis=0)
    vis = np.concatenate([gy, gx], -1).reshape(ph * pw, D_AXIS)
    cos_vis = np.cos(vis)
    sin_vis = np.sin(vis)
    t = np.arange(MAX_TEXT, dtype=np.float32)
    tf = np.outer(t, freqs)
    tf = np.concatenate([tf, tf], -1)
    cos_t = np.cos(tf)
    sin_t = np.sin(tf)
    cos = np.ones((1, TOTAL, D_AXIS), dtype=np.float32)
    sin = np.zeros((1, TOTAL, D_AXIS), dtype=np.float32)
    cos[0, : ph * pw] = cos_vis
    sin[0, : ph * pw] = sin_vis
    cos[0, MAX_PATCHES : MAX_PATCHES + MAX_TEXT] = cos_t
    sin[0, MAX_PATCHES : MAX_PATCHES + MAX_TEXT] = sin_t
    return cos, sin


def make_prefill_mask() -> np.ndarray:
    """Create mask for prefill (1,1,320,320)."""
    mask = np.full((TOTAL, TOTAL), -1e9, dtype=np.float32)
    mask[:MAX_PATCHES, :MAX_PATCHES] = 0
    mask[:MAX_PATCHES, MAX_PATCHES:] = -1e9
    mask[MAX_PATCHES : MAX_PATCHES + 1, :MAX_PATCHES] = 0
    mask[MAX_PATCHES : MAX_PATCHES + 1, MAX_PATCHES : MAX_PATCHES + 1] = 0
    mask[MAX_PATCHES : MAX_PATCHES + 1, MAX_PATCHES + 1 :] = -1e9
    mask[MAX_PATCHES + 1 :, : MAX_PATCHES] = 0
    mask[MAX_PATCHES + 1 :, MAX_PATCHES:] = -1e9
    return mask.reshape(1, 1, TOTAL, TOTAL)


def make_decode_mask(pos: int) -> np.ndarray:
    """Create decode mask (1,1,1,320) for current pos."""
    m = np.full((1, 1, 1, TOTAL), -1e9, dtype=np.float32)
    m[0, 0, 0, : pos + 1] = 0
    return m


# HF download helpers
def _resolve_litert_dir(
    quant: str = "wi4",
    hf_repo: str = DEFAULT_HF_REPO,
    local_dir: Optional[Union[str, Path]] = None,
    revision: Optional[str] = None,
) -> Path:
    """Resolve local directory containing LiteRT artefacts for given quant.

    If ``local_dir`` is provided and exists, it is used directly (expects the
    quant folder layout or directly the tflite files). Otherwise downloads from
    HF using ``huggingface_hub``.

    Returns Path to the quant folder.
    """
    canon = normalize_litert_quant(quant)

    if local_dir is not None:
        p = Path(local_dir)
        if p.is_dir():
            if list(p.glob("*.tflite")):
                return p
            cand = p / canon
            if cand.is_dir() and list(cand.glob("*.tflite")):
                return cand
            return p
        if p.is_file():
            return p.parent
        logger.warning(f"litert_model_path {p} does not exist, falling back to HF download")

    # Download from HF
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for LiteRT backend auto-download. "
            "Install with `pip install huggingface_hub` or provide a local "
            "`litert_model_path` containing the exported tflite files."
        ) from e

    pattern = f"litert_exports/{canon}/*"
    logger.info(f"Downloading LiteRT artefacts ({canon}) from {hf_repo} ...")
    snapshot_dir = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=[pattern],
            revision=revision,
        )
    )
    quant_dir = snapshot_dir / "litert_exports" / canon
    if not quant_dir.is_dir():
        cands = list(snapshot_dir.rglob("position_base.npy"))
        for c in cands:
            if canon in str(c):
                return c.parent
        raise FileNotFoundError(f"Failed to locate LiteRT quant folder {canon} in {snapshot_dir}")
    return quant_dir


def _find_tflite_files(quant_dir: Path, quant: str) -> dict:
    canon = normalize_litert_quant(quant)
    suffix = QUANT_SUFFIX[canon]
    enc_cands = list(quant_dir.glob(f"hayai_encoder_{suffix}.tflite"))
    pre_cands = list(quant_dir.glob(f"hayai_decoder_prefill_{suffix}.tflite"))
    dec_cands = list(quant_dir.glob(f"hayai_decoder_decode_{suffix}.tflite"))
    if not enc_cands:
        enc_cands = list(quant_dir.glob("hayai_encoder*.tflite"))
    if not pre_cands:
        pre_cands = list(quant_dir.glob("hayai_decoder_prefill*.tflite"))
    if not dec_cands:
        dec_cands = list(quant_dir.glob("hayai_decoder_decode*.tflite"))
    if not (enc_cands and pre_cands and dec_cands):
        raise FileNotFoundError(
            f"Missing tflite files in {quant_dir}. Found: "
            f"encoder={enc_cands} prefill={pre_cands} decode={dec_cands}"
        )
    return {"encoder": enc_cands[0], "prefill": pre_cands[0], "decode": dec_cands[0]}


# LiteRT engine
class LitertOcrEngine:
    """LiteRT KV-cache inference engine (no torch required)."""

    def __init__(
        self,
        quant: str = "wi4",
        model_dir: Optional[Union[str, Path]] = None,
        hf_repo: str = DEFAULT_HF_REPO,
        revision: Optional[str] = None,
        num_threads: Optional[int] = None,
    ):
        """
        :param quant: quantization preset. One of ``none``, ``wi4``, ``wi8_afp32``,
            ``dynamic_wi4``, ``dynamic_wi8`` (aliases ``int4``, ``int8``, ``float`` etc).
        :param model_dir: local path to quant folder (or litert_exports root). If None,
            downloads from ``hf_repo``.
        :param hf_repo: HF repository id for auto-download.
        :param revision: HF revision.
        :param num_threads: number of threads for LiteRT interpreters; defaults to system.
        """
        self.quant = normalize_litert_quant(quant)
        self.hf_repo = hf_repo
        self.quant_dir = _resolve_litert_dir(quant=self.quant, hf_repo=hf_repo, local_dir=model_dir, revision=revision)

        files = _find_tflite_files(self.quant_dir, self.quant)
        self.encoder_path = files["encoder"]
        self.prefill_path = files["prefill"]
        self.decode_path = files["decode"]

        position_base_path = self.quant_dir / "position_base.npy"
        tokenizer_path = self.quant_dir / "tokenizer.json"
        if not position_base_path.is_file():
            raise FileNotFoundError(f"Missing position_base.npy in {self.quant_dir}")
        if not tokenizer_path.is_file():
            tokenizer_path = Path("tokenizer.json")
            if not tokenizer_path.is_file():
                raise FileNotFoundError(f"Missing tokenizer.json in {self.quant_dir}")

        self.position_base = np.load(str(position_base_path))

        try:
            import ai_edge_litert.interpreter as litert  # type: ignore
        except ImportError as e:
            raise ImportError(
                "ai_edge_litert is required for LiteRT backend. "
                "Install with `pip install ai-edge-litert` (or `pip install hayai-ocr[litert]`)."
            ) from e
        try:
            from tokenizers import Tokenizer  # type: ignore
        except ImportError as e:
            raise ImportError("tokenizers is required for LiteRT backend. `pip install tokenizers`.") from e

        self._litert = litert
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.bos_id = self.tokenizer.token_to_id("<bos>")
        self.eos_id = self.tokenizer.token_to_id("<eos>")
        self.pad_id = self.tokenizer.token_to_id("<pad>")
        if self.bos_id is None:
            self.bos_id = 1
        if self.eos_id is None:
            self.eos_id = 2
        if self.pad_id is None:
            self.pad_id = self.eos_id

        logger.info(f"Loading LiteRT interpreters (quant={self.quant}) from {self.quant_dir}")
        self.enc = litert.Interpreter(model_path=str(self.encoder_path), num_threads=num_threads)
        self.enc.allocate_tensors()
        self.prefill = litert.Interpreter(model_path=str(self.prefill_path), num_threads=num_threads)
        self.prefill.allocate_tensors()
        self.decode = litert.Interpreter(model_path=str(self.decode_path), num_threads=num_threads)
        self.decode.allocate_tensors()

        self._enc_in = [d["index"] for d in self.enc.get_input_details()]
        self._enc_out = [d["index"] for d in self.enc.get_output_details()]
        self._pre_in = [d["index"] for d in self.prefill.get_input_details()]
        self._pre_out = [d["index"] for d in self.prefill.get_output_details()]
        self._dec_in = [d["index"] for d in self.decode.get_input_details()]
        self._dec_out = [d["index"] for d in self.decode.get_output_details()]

        logger.info(f"LiteRT backend ready (encoder={self.encoder_path.name}, prefill={self.prefill_path.name})")

    def generate(
        self,
        pil_image: Image.Image,
        max_new_tokens: int = 64,
        repetition_penalty: float = 1.00,
    ) -> str:
        """Generate text for a single PIL image (greedy)."""
        pv, amask, (ph, pw) = preprocess_image_pil(pil_image)
        pos_embeds = compute_pos_embeds(self.position_base, ph, pw)
        cos, sin = compute_rope(ph, pw)

        # Encoder
        self.enc.set_tensor(self._enc_in[0], pv.astype(np.float32))
        self.enc.set_tensor(self._enc_in[1], amask.astype(np.int64))
        self.enc.set_tensor(self._enc_in[2], pos_embeds.astype(np.float32))
        self.enc.invoke()
        visual = self.enc.get_tensor(self._enc_out[0])  # (1,256,768)

        mask_prefill = make_prefill_mask()
        text_ids = np.array([[self.bos_id]], dtype=np.int64)

        self.prefill.set_tensor(self._pre_in[0], visual.astype(np.float32))
        self.prefill.set_tensor(self._pre_in[1], text_ids)
        self.prefill.set_tensor(self._pre_in[2], cos.astype(np.float32))
        self.prefill.set_tensor(self._pre_in[3], sin.astype(np.float32))
        self.prefill.set_tensor(self._pre_in[4], mask_prefill.astype(np.float32))
        self.prefill.invoke()
        logits = self.prefill.get_tensor(self._pre_out[0])
        caches = [self.prefill.get_tensor(idx) for idx in self._pre_out[1:]]

        if repetition_penalty != 1.0:
            if logits[0, self.bos_id] < 0:
                logits[0, self.bos_id] *= repetition_penalty
            else:
                logits[0, self.bos_id] /= repetition_penalty

        next_id = int(np.argmax(logits[0]))
        generated: List[int] = []
        if next_id not in (self.eos_id, self.pad_id):
            generated.append(next_id)
        else:
            return self.tokenizer.decode(generated, skip_special_tokens=True)

        seen = [self.bos_id] + generated

        for step in range(1, max_new_tokens):
            if next_id in (self.eos_id, self.pad_id):
                break
            if len(generated) >= max_new_tokens:
                break
            cur_pos = MAX_PATCHES + len(generated)
            token_ids = np.array([[next_id]], dtype=np.int64)
            cos_step = cos[:, cur_pos : cur_pos + 1, :].astype(np.float32)
            sin_step = sin[:, cur_pos : cur_pos + 1, :].astype(np.float32)
            mask_decode = make_decode_mask(cur_pos).astype(np.float32)

            self.decode.set_tensor(self._dec_in[0], token_ids)
            self.decode.set_tensor(self._dec_in[1], cos_step)
            self.decode.set_tensor(self._dec_in[2], sin_step)
            self.decode.set_tensor(self._dec_in[3], mask_decode)
            self.decode.set_tensor(self._dec_in[4], np.array(cur_pos, dtype=np.int64))
            for i, c in enumerate(caches):
                self.decode.set_tensor(self._dec_in[5 + i], c)
            self.decode.invoke()
            logits = self.decode.get_tensor(self._dec_out[0])
            caches = [self.decode.get_tensor(idx) for idx in self._dec_out[1:]]

            if repetition_penalty != 1.0 and len(seen) > 0:
                for tok_id in set(seen):
                    val = logits[0, tok_id]
                    if val < 0:
                        logits[0, tok_id] = val * repetition_penalty
                    else:
                        logits[0, tok_id] = val / repetition_penalty

            next_id = int(np.argmax(logits[0]))
            if next_id in (self.eos_id, self.pad_id):
                break
            generated.append(next_id)
            seen.append(next_id)

        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def generate_batch(
        self,
        pil_images: List[Image.Image],
        max_new_tokens: int = 64,
        repetition_penalty: float = 1.00,
    ) -> List[str]:
        """Generate for a batch (loops sequentially, LiteRT batch=1)."""
        return [self.generate(img, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty) for img in pil_images]

