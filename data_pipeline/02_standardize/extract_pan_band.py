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

# ITU-R BT.709 luma weights (matches sRGB-class sensors better than BT.601)
LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)


def extract_true_pan(src: rasterio.DatasetReader, band_index: int = 1):
    if src.count != 1 and band_index > src.count:
        raise ValueError(f"band_index={band_index} out of range for {src.count}-band source")
    data = src.read(band_index)
    tags = {"band_source": "true_pan", "pseudo_pan": "false", "source_band_index": str(band_index)}
    return data, tags


def extract_pseudo_pan(src: rasterio.DatasetReader):
    if src.count < 3:
        raise ValueError(f"rgb_to_pseudo_pan needs >= 3 bands, source has {src.count}")
    src_dtype = np.dtype(src.dtypes[0])
    rgb = src.read([1, 2, 3]).astype(np.float64)
    luminance = sum(w * band for w, band in zip(LUMA_WEIGHTS, rgb))

    if np.issubdtype(src_dtype, np.floating):
        data = np.clip(luminance, 0.0, 1.0).astype(src_dtype)
    else:
        data = np.clip(luminance, 0, np.iinfo(src_dtype).max).astype(src_dtype)

    tags = {"band_source": "rgb_luminance_bt709", "pseudo_pan": "true"}
    return data, tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    ap.add_argument("--mode", required=True, choices=["true_pan", "rgb_to_pseudo_pan"])
    ap.add_argument("--band-index", type=int, default=1, help="for true_pan: which band is PAN (1-indexed)")
    args = ap.parse_args()

    with rasterio.open(args.in_path) as src:
        if args.mode == "true_pan":
            data, extra_tags = extract_true_pan(src, args.band_index)
        else:
            data, extra_tags = extract_pseudo_pan(src)

        profile = src.profile.copy()
        profile.update(count=1, dtype=data.dtype)

        args.out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(args.out_path, "w", **profile) as dst:
            dst.write(data, 1)
            dst.update_tags(**extra_tags, source_file=args.in_path.name)

    print(f"{args.in_path.name} -> {args.out_path.name}  "
          f"mode={args.mode}  shape={data.shape}  dtype={data.dtype}  "
          f"range=[{data.min()}, {data.max()}]")


if __name__ == "__main__":
    main()
