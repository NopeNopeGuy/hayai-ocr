"""ONNX Runtime backend for Hayai OCR v2.7 Supernova.

Runs exported KV-cache prefill + decode ONNX graphs via ``onnxruntime`` — no PyTorch required.
Artefacts: https://huggingface.co/JustANormalTinkerer/hayai-ocr-v2.7-supernova-onnx
Precisions: fp32, fp16, int8 (default), int4.

Usage:
    HayaiOcr(backend="onnx", onnx_precision="int8")
    HayaiONNXPipeline(precision="int8").ocr(image)

Requires: onnxruntime, tokenizers, huggingface_hub.
"""
from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnxruntime as ort
from PIL import Image
from loguru import logger
from tokenizers import Tokenizer

DEFAULT_HF_REPO = "JustANormalTinkerer/hayai-ocr-v2.7-supernova-onnx"

CANONICAL_PRECISIONS = {"fp32", "fp16", "int8", "int4"}
PRECISION_ALIASES: Dict[str, str] = {
    "fp32": "fp32",
    "float": "fp32",
    "float32": "fp32",
    "none": "fp32",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "int8": "int8",
    "wi8": "int8",
    "i8": "int8",
    "int4": "int4",
    "wi4": "int4",
    "i4": "int4",
}


def normalize_onnx_precision(precision: Optional[str]) -> str:
    """Normalize user supplied precision string to canonical name."""
    if precision is None:
        return "int8"
    p = precision.strip().lower()
    if p in PRECISION_ALIASES:
        return PRECISION_ALIASES[p]
    raise ValueError(
        f"Unknown onnx_precision {precision!r}. Valid choices: {sorted(CANONICAL_PRECISIONS)} "
        f"plus aliases {sorted(PRECISION_ALIASES)}"
    )


def _resolve_onnx_dir(
    precision: str = "int8",
    hf_repo: str = DEFAULT_HF_REPO,
    local_dir: Optional[Union[str, Path]] = None,
    revision: Optional[str] = None,
) -> Path:
    canon = normalize_onnx_precision(precision)
    suffix = "" if canon == "fp32" else f"_{canon}"
    prefill_name = f"hayai_prefill{suffix}.onnx"
    decode_name = f"hayai_decode{suffix}.onnx"

    if local_dir is not None:
        p = Path(local_dir)
        if p.is_dir():
            return p
        if p.is_file():
            return p.parent
        logger.warning(f"onnx_model_path {p} does not exist, falling back to HF download")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for ONNX backend auto-download. "
            "Install with `pip install huggingface_hub` or provide a local "
            "`onnx_model_path` containing the exported .onnx files."
        ) from e

    allow_patterns = [
        "hayai_onnx_config.json",
        "pos_embedding.npy",
        "tokenizer.json",
        "tokenizer_config.json",
        prefill_name,
        decode_name,
    ]
    logger.info(f"Downloading ONNX artefacts ({canon}) from {hf_repo} ...")
    snapshot_dir = Path(
        snapshot_download(
            repo_id=hf_repo,
            allow_patterns=allow_patterns,
            revision=revision,
        )
    )
    return snapshot_dir


@lru_cache(maxsize=16384)
def _naflex_target_size(
    height: int,
    width: int,
    max_patches: int = 256,
    patch_size: int = 16,
    min_patches: int = 64,
    max_scale: float = 4.0,
) -> Tuple[int, int, int, int]:
    scale_min, scale_max = 1e-5, max_scale
    while (scale_max - scale_min) >= 1e-5:
        scale = 0.5 * (scale_min + scale_max)
        th = max(patch_size, int(math.ceil(height * scale / patch_size)) * patch_size)
        tw = max(patch_size, int(math.ceil(width * scale / patch_size)) * patch_size)
        if (th // patch_size) * (tw // patch_size) <= max_patches:
            scale_min = scale
        else:
            scale_max = scale

    th = max(patch_size, int(math.ceil(height * scale_min / patch_size)) * patch_size)
    tw = max(patch_size, int(math.ceil(width * scale_min / patch_size)) * patch_size)
    nh, nw = th // patch_size, tw // patch_size

    if nh * nw < min_patches:
        s_min, s_max = scale_min, 100.0
        target_min = min(min_patches, max_patches)
        while (s_max - s_min) >= 1e-5:
            s = 0.5 * (s_min + s_max)
            th = max(patch_size, int(math.ceil(height * s / patch_size)) * patch_size)
            tw = max(patch_size, int(math.ceil(width * s / patch_size)) * patch_size)
            if (th // patch_size) * (tw // patch_size) <= target_min:
                s_min = s
            else:
                s_max = s
        th = max(patch_size, int(math.ceil(height * s_min / patch_size)) * patch_size)
        tw = max(patch_size, int(math.ceil(width * s_min / patch_size)) * patch_size)
        nh, nw = th // patch_size, tw // patch_size

    return th, tw, nh, nw


@lru_cache(maxsize=16384)
def _select_best_max_patches(
    height: int,
    width: int,
    patch_size: int = 16,
    min_patches: int = 64,
    max_scale: float = 4.0,
) -> Tuple[int, int, int]:
    info: Dict[int, Tuple[int, float, int]] = {}
    for mp in (128, 192, 256, 384):
        th, tw, nh, nw = _naflex_target_size(height, width, mp, patch_size, min_patches, max_scale)
        m_vis = ((nh + 1) // 2) * ((nw + 1) // 2)
        info[mp] = (nh * nw, min(th / max(1, height), tw / max(1, width)), m_vis)

    n128, s128, m128 = info[128]
    n192, s192, m192 = info[192]
    n256, s256, m256 = info[256]
    n384, _, m384 = info[384]

    if n128 == n384:
        return 128, n128, m128

    h_safe, w_safe = max(1, height), max(1, width)
    aspect = max(h_safe, w_safe) / min(h_safe, w_safe)
    w_over_h = w_safe / h_safe
    is_square = 0.77 <= w_over_h <= 1.30
    is_horizontal = w_over_h > 1.30

    if not is_square:
        if not is_horizontal and s128 >= 1.65 and aspect < 3.8:
            return 128, n128, m128
        if is_horizontal and s128 >= 2.35 and aspect < 3.2:
            return 128, n128, m128

    if n384 > n256:
        if (
            s256 < 1.28
            or (is_square and s256 < 2.20)
            or (is_horizontal and s256 < 1.45)
            or (aspect >= 3.8 and s256 < 1.75)
            or (aspect >= 5.0 and s256 < 2.20)
        ):
            return 384, n384, m384

    if n192 == n128:
        return 128, n128, m128
    if n256 == n192:
        return 192, n192, m192
    if not is_square:
        if not is_horizontal and s192 >= 1.55 and aspect < 3.6:
            return 192, n192, m192
        if is_horizontal and s192 >= 1.85 and aspect < 3.0:
            return 192, n192, m192

    return 256, n256, m256


class HayaiONNXPipeline:
    def __init__(
        self,
        model_dir: Optional[Union[str, Path]] = None,
        precision: str = "int8",  # "fp32", "fp16", "int8", or "int4"
        hf_repo: str = DEFAULT_HF_REPO,
        revision: Optional[str] = None,
        providers: Optional[List[str]] = None,
        num_threads: Optional[int] = None,
    ):
        self.precision = normalize_onnx_precision(precision)
        self.hf_repo = hf_repo
        d = _resolve_onnx_dir(
            precision=self.precision,
            hf_repo=hf_repo,
            local_dir=model_dir,
            revision=revision,
        )
        self.model_dir = d

        self.meta = json.loads((d / "hayai_onnx_config.json").read_text())
        self.tokenizer = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.pos_w = np.load(d / "pos_embedding.npy")
        self.pos_side = int(round(self.pos_w.shape[0] ** 0.5))
        self.pos_grid = self.pos_w.reshape(self.pos_side, self.pos_side, -1).astype(np.float32)

        if providers is None:
            avail = ort.get_available_providers()
            if self.precision in ("int8", "int4") or "CUDAExecutionProvider" not in avail:
                providers = ["CPUExecutionProvider"]
            else:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if num_threads is not None:
            sess_opts.intra_op_num_threads = int(num_threads)
            sess_opts.inter_op_num_threads = 1
        elif providers == ["CPUExecutionProvider"]:
            sess_opts.intra_op_num_threads = os.cpu_count() or 4
            sess_opts.inter_op_num_threads = 1

        suffix = "" if self.precision == "fp32" else f"_{self.precision}"
        prefill_path = d / f"hayai_prefill{suffix}.onnx"
        decode_path = d / f"hayai_decode{suffix}.onnx"
        if not prefill_path.is_file() or not decode_path.is_file():
            raise FileNotFoundError(
                f"Missing ONNX files for precision={self.precision!r} in {d}: "
                f"{prefill_path.name}, {decode_path.name}"
            )

        logger.info(
            f"Loading ONNX sessions (precision={self.precision}, providers={providers}) from {d}"
        )
        self.prefill_sess = ort.InferenceSession(str(prefill_path), sess_opts, providers=providers)
        self.decode_sess = ort.InferenceSession(str(decode_path), sess_opts, providers=providers)

    @lru_cache(maxsize=512)
    def _pos_embed(self, nh: int, nw: int) -> np.ndarray:
        def _aa_1d(out_sz: int, in_sz: int) -> np.ndarray:
            scale = float(in_sz) / float(out_sz)
            invscale = (1.0 / scale) if scale > 1.0 else 1.0
            centers = scale * (np.arange(out_sz, dtype=np.float32) + 0.5)
            j = np.arange(in_sz, dtype=np.float32) + 0.5
            w = np.maximum(0.0, 1.0 - np.abs(j[None, :] - centers[:, None]) * invscale)
            return w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)

        wh = _aa_1d(nh, self.pos_side)
        ww = _aa_1d(nw, self.pos_side)
        return np.einsum("ih,jw,hwd->ijd", wh, ww, self.pos_grid).reshape(nh * nw, -1)

    def _preprocess(self, images: List[Image.Image], max_patches: Union[int, str] = "auto"):
        ps = self.meta["patch_size"]
        min_p = self.meta["min_patches"]
        ms = self.meta["max_scale"]
        mult = self.meta["pad_to_multiple_of"]
        patch_dim = ps * ps * 3

        decoded = []
        for im in images:
            rgb = im.convert("RGB") if im.mode != "RGB" else im
            w, h = rgb.size
            mp = _select_best_max_patches(h, w, ps, min_p, ms)[0] if max_patches == "auto" else int(max_patches)
            th, tw, nh, nw = _naflex_target_size(h, w, mp, ps, min_p, ms)
            if (tw, th) != (w, h):
                rgb = rgb.resize((tw, th), resample=Image.Resampling.BILINEAR)
            arr = np.asarray(rgb, dtype=np.uint8)
            patches = arr.reshape(nh, ps, nw, ps, 3).transpose(0, 2, 1, 3, 4).reshape(nh * nw, patch_dim)
            decoded.append((patches, nh, nw))

        b = len(decoded)
        max_valid = max(nh * nw for _, nh, nw in decoded)
        n_keep = max(mult, ((max_valid + mult - 1) // mult) * mult)

        pv = np.zeros((b, n_keep, patch_dim), dtype=np.float32)
        pos_embeds = np.zeros((b, n_keep, self.pos_w.shape[1]), dtype=np.float32)
        spatial_shapes = np.empty((b, 2), dtype=np.int64)

        for j, (patches, nh, nw) in enumerate(decoded):
            np_j = nh * nw
            pv[j, :np_j] = patches.astype(np.float32) * (2.0 / 255.0) - 1.0
            pos_embeds[j, :np_j] = self._pos_embed(nh, nw)
            spatial_shapes[j] = (nh, nw)

        ho = (spatial_shapes[:, 0] + 1) // 2
        wo = (spatial_shapes[:, 1] + 1) // 2
        l_vis = int(np.max(ho * wo))
        l_vis_arange = np.arange(l_vis, dtype=np.int64)
        return pv, pos_embeds, spatial_shapes, l_vis_arange

    def _run_batch(self, images: List[Image.Image], max_patches: Union[int, str], max_new_tokens: int) -> List[str]:
        b = len(images)
        bos = self.meta["bos_token_id"]
        eos = self.meta["eos_token_id"]
        pad = self.meta["pad_token_id"]

        pv, pos_embeds, spatial_shapes, l_vis_arange = self._preprocess(images, max_patches=max_patches)
        bos_ids = np.full((b, 1), bos, dtype=np.int64)

        logits, k_cache, v_cache, vis_valid = self.prefill_sess.run(
            None,
            {
                "pixel_values": pv,
                "pos_embeds": pos_embeds,
                "spatial_shapes": spatial_shapes,
                "bos_ids": bos_ids,
                "l_vis_arange": l_vis_arange,
            },
        )

        nxt = np.argmax(logits, axis=-1).astype(np.int64)
        unfinished = (nxt != eos) & (nxt != pad)
        sequences = [[int(t)] for t in nxt]
        past_attn_mask = np.concatenate([vis_valid, np.ones((b, 1), dtype=np.int64)], axis=1)

        for step in range(1, max_new_tokens):
            if not np.any(unfinished):
                break
            logits, k_cache, v_cache = self.decode_sess.run(
                None,
                {
                    "input_ids": nxt[:, None],
                    "past_attention_mask": past_attn_mask,
                    "position_id": np.array([step], dtype=np.int64),
                    "k_cache_in": k_cache,
                    "v_cache_in": v_cache,
                },
            )
            past_attn_mask = np.concatenate([past_attn_mask, np.ones((b, 1), dtype=np.int64)], axis=1)
            nxt = np.where(unfinished, np.argmax(logits, axis=-1).astype(np.int64), pad)
            for j in range(b):
                if unfinished[j]:
                    sequences[j].append(int(nxt[j]))
            unfinished &= (nxt != eos) & (nxt != pad)

        return [
            self.tokenizer.decode([t for t in seq if t not in (bos, eos, pad)], skip_special_tokens=True)
            for seq in sequences
        ]

    def ocr(
        self,
        images: Union[Image.Image, List[Image.Image]],
        max_patches: Union[int, str] = "auto",
        max_new_tokens: int = 128,
        batch_size: int = 32,
    ) -> Union[str, List[str]]:
        single = not isinstance(images, (list, tuple))
        img_list = [images] if single else list(images)

        if not single and max_patches == "auto" and len(img_list) > 1:
            ps, min_p, ms = self.meta["patch_size"], self.meta["min_patches"], self.meta["max_scale"]
            buckets: Dict[int, List[Tuple[int, int, int]]] = {128: [], 192: [], 256: [], 384: []}
            for idx, im in enumerate(img_list):
                w, h = im.size
                mp, npatch, m_vis = _select_best_max_patches(h, w, ps, min_p, ms)
                buckets[mp].append((idx, m_vis, npatch))

            results = [""] * len(img_list)
            for mp in (128, 192, 256, 384):
                items = sorted(buckets[mp], key=lambda x: (x[1], x[2]))
                for start in range(0, len(items), batch_size):
                    chunk = items[start : start + batch_size]
                    preds = self._run_batch([img_list[c[0]] for c in chunk], mp, max_new_tokens)
                    for (orig_idx, _, _), pred in zip(chunk, preds):
                        results[orig_idx] = pred
            return results

        if not single and len(img_list) > batch_size:
            results = []
            for start in range(0, len(img_list), batch_size):
                results.extend(self._run_batch(img_list[start : start + batch_size], max_patches, max_new_tokens))
            return results

        out = self._run_batch(img_list, max_patches, max_new_tokens)
        return out[0] if single else out


OnnxOcrEngine = HayaiONNXPipeline
