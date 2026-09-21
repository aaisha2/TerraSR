"""Stage 2a: extract a single-band PAN (or pseudo-PAN) image from a raw
downloaded scene, before CRS/dtype standardization (to_geotiff.py).

Two modes, matching the tiers in the build plan (docs/plan §2):

  true_pan          -- source already has a real panchromatic band
                       (SpaceNet PAN/*.TIF, Maxar pan_analytic). Pass the
                       band through unchanged; this is the primary training
                       signal and must never be altered radiometrically.

  rgb_to_pseudo_pan -- source is RGB only (OpenEarthMap, NAIP). Compute a
                       luminance-weighted single-band proxy (ITU-R BT.709
                       weights) and tag the output `pseudo_pan=true` so it's
                       never conflated with true PAN downstream (confirmed
                       requirement from supervisor, 2026-07-11: pseudo-PAN
                       is supplementary scene-diversity data only, never a
                       substitute for true PAN in the primary training mix).

Usage:
    python extract_pan_band.py --in raw.tif --out out.tif --mode true_pan
    python extract_pan_band.py --in raw_rgb.tif --out out.tif --mode rgb_to_pseudo_pan
"""
import argparse
from pathlib import Path

import numpy as np
import rasterio

from _common import finalize_atomic, row_windows, tmp_path_for

# ITU-R BT.709 luma weights (matches sRGB-class sensors better than BT.601)
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)

TRUE_PAN_TAGS = {"band_source": "true_pan", "pseudo_pan": "false"}
PSEUDO_PAN_TAGS = {"band_source": "rgb_luminance_bt709", "pseudo_pan": "true"}


def extract_true_pan(src: rasterio.DatasetReader, band_index: int = 1, window=None):
    if band_index > src.count:
        raise ValueError(f"band_index={band_index} out of range for {src.count}-band source")
    data = src.read(band_index, window=window)
    tags = dict(TRUE_PAN_TAGS, source_band_index=str(band_index))
    return data, tags


def extract_pseudo_pan(src: rasterio.DatasetReader, window=None):
    if src.count < 3:
        raise ValueError(f"rgb_to_pseudo_pan needs >= 3 bands, source has {src.count}")
    src_dtype = np.dtype(src.dtypes[0])
    rgb = src.read([1, 2, 3], window=window).astype(np.float64)
    luminance = sum(w * band for w, band in zip(LUMA_WEIGHTS, rgb))

    if np.issubdtype(src_dtype, np.floating):
        data = np.clip(luminance, 0.0, 1.0).astype(src_dtype)
    else:
        data = np.clip(luminance, 0, np.iinfo(src_dtype).max).astype(src_dtype)
    return data, dict(PSEUDO_PAN_TAGS)


def extract_pan_to_file(in_path: Path, out_path: Path, mode: str, band_index: int = 1) -> dict:
    """Extract the PAN/pseudo-PAN band from in_path and write it to out_path.
    Reusable by the batch driver (standardize_scenes.py) and the CLI below.

    Processes the scene in row strips, so memory stays bounded for scenes of
    any size; the output appears only once complete (atomic rename)."""
    with rasterio.open(in_path) as src:
        if mode == "true_pan" and band_index > src.count:
            raise ValueError(f"band_index={band_index} out of range for {src.count}-band source")
        if mode != "true_pan" and src.count < 3:
            raise ValueError(f"rgb_to_pseudo_pan needs >= 3 bands, source has {src.count}")

        profile = src.profile.copy()
        profile.update(count=1, dtype=src.dtypes[0], tiled=True,
                       blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
        profile.pop("photometric", None)   # RGB photometric is invalid for 1 band

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tmp_path_for(out_path)
        lo, hi, tags = None, None, {}
        with rasterio.open(tmp, "w", **profile) as dst:
            for win in row_windows(src.width, src.height):
                if mode == "true_pan":
                    data, tags = extract_true_pan(src, band_index, window=win)
                else:
                    data, tags = extract_pseudo_pan(src, window=win)
                dst.write(data, 1, window=win)
                lo = data.min() if lo is None else min(lo, data.min())
                hi = data.max() if hi is None else max(hi, data.max())
            dst.update_tags(**tags, source_file=in_path.name)
        finalize_atomic(tmp, out_path)

    return {"shape": (src.height, src.width), "dtype": str(src.dtypes[0]),
            "min": float(lo), "max": float(hi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--mode", required=True, choices=["true_pan", "rgb_to_pseudo_pan"])
    ap.add_argument("--band-index", type=int, default=1, help="for true_pan: which band is PAN (1-indexed)")
    args = ap.parse_args()

    info = extract_pan_to_file(args.in_path, args.out_path, args.mode, args.band_index)
    print(f"{args.in_path.name} -> {args.out_path.name}  mode={args.mode}  "
          f"shape={info['shape']}  dtype={info['dtype']}  "
          f"range=[{info['min']}, {info['max']}]")


if __name__ == "__main__":
    main()
