"""I/O helpers for stage 5 — GeoTIFF (real pipeline, 16-bit + georeference
preserving) and PNG (legacy 8-bit smoke-test) read/write.

HR is always written back losslessly (the original array + profile, unchanged
dtype/transform/CRS/tags) — it's the ground truth and must not be re-encoded.
Only the LR is synthesized: normalized to [0,1] for the degradation math,
degraded, then mapped back to the source dtype so HR and LR share one
radiometric scale, and written with a geotransform scaled by the SR factor so
the LR stays correctly georeferenced.
"""
import os
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio import Affine


def dtype_max(dtype) -> float:
    dt = np.dtype(dtype)
    if np.issubdtype(dt, np.integer):
        return float(np.iinfo(dt).max)
    return 1.0


# ---- GeoTIFF ----------------------------------------------------------------

def read_geotiff(path):
    """Return (array, profile, tags) for a single-band GeoTIFF."""
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        tags = src.tags()
    return arr, profile, tags


def normalize_for_degradation(arr: np.ndarray, mode: str) -> tuple:
    """Map a native-dtype patch to float [0,1] for the degradation pipeline.
    Returns (normalized_float, norm_scale) where norm_scale denormalizes the
    degraded LR back to the source dtype's scale.

    per_patch_max: divide by the patch's own max — keeps the [0,1]-relative
    noise parameters meaningful even for PAN patches that only fill a fraction
    of the dtype range (common: 12-bit sensor data stored in uint16).
    dtype_max: divide by the full dtype range — preserves absolute radiometry
    across patches but makes noise params under-act on under-filled patches."""
    arr_f = arr.astype(np.float64)
    if mode == "dtype_max":
        scale = dtype_max(arr.dtype)
    elif mode == "per_patch_max":
        scale = float(arr.max()) or 1.0
    else:
        raise ValueError(f"unknown normalization mode: {mode}")
    return np.clip(arr_f / scale, 0.0, 1.0), scale


def denormalize_lr(lr_float: np.ndarray, norm_scale: float, dtype) -> np.ndarray:
    dt = np.dtype(dtype)
    hi = dtype_max(dt)
    out = np.clip(lr_float * norm_scale, 0, hi)
    return out.astype(dt) if np.issubdtype(dt, np.integer) else out.astype(dt)


def scaled_transform(src_transform: Affine, scale_factor: int) -> Affine:
    """LR pixels are `scale_factor`x larger on the ground; same origin."""
    return src_transform * Affine.scale(scale_factor)


def _standard_profile(src_profile: dict, arr: np.ndarray, transform) -> dict:
    profile = src_profile.copy()
    profile.update(
        driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
        dtype=arr.dtype, transform=transform, compress="deflate", predictor=2,
        tiled=True, blockxsize=256, blockysize=256,
    )
    return profile


def _tmp_for(path) -> Path:
    path = Path(path)
    return path.with_name(path.stem + ".partial" + path.suffix)


def write_geotiff(path, arr: np.ndarray, src_profile: dict, transform, tags: dict):
    """Written to a temp name and renamed when complete, so a pair file that
    exists is always a finished one (stage 5 resume relies on this)."""
    profile = _standard_profile(src_profile, arr, transform)
    tmp = _tmp_for(path)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr, 1)
        if tags:
            dst.update_tags(**tags)
    os.replace(tmp, path)


# ---- PNG (legacy 8-bit smoke-test path) -------------------------------------

def read_png_gray_float(path) -> np.ndarray:
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float64) / 255.0


def write_png_gray_float(arr: np.ndarray, path) -> None:
    img = Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8))
    tmp = _tmp_for(path)
    img.save(tmp, format="PNG")
    os.replace(tmp, path)
