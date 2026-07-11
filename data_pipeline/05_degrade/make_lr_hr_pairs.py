"""Stage 5 CLI: turn HR patches into LR/HR pairs using the confirmed
single-order degradation pipeline (configs/degradation.yaml).

Two output formats (configs/degradation.yaml -> output.format):
  geotiff  real pipeline — preserves 16-bit depth + georeferencing. HR is
           copied losslessly; LR is written as GeoTIFF with a geotransform
           scaled by the SR factor. Pass GeoTIFF HR patches (stage 3/4 output).
  png      legacy 8-bit path for the synthetic smoke-test images only.

Two ways to select the HR patches:
  --manifest  a stage 4 labeled manifest — processes only kept, labeled
              patches (patch_path as HR, tile_id as id). Preferred for the
              real pipeline so stage 5 output aligns exactly with stage 6.
  --hr-dir    a directory of HR images — processes every image in it
              (id = filename stem). Used for the synthetic smoke test.

Usage:
    python make_lr_hr_pairs.py --hr-dir tests/sample_images --out-dir out/pairs
    python make_lr_hr_pairs.py --manifest out/patches/patch_manifest_labeled.json --out-dir out/pairs
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from degradation_pipeline import degrade  # noqa: E402
import patch_io as pio  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def collect_hr_items(args):
    """Yield (id, hr_source_path) pairs from either --manifest or --hr-dir."""
    if args.manifest:
        manifest = json.loads(args.manifest.read_text())
        for row in manifest:
            if row.get("keep") is False:
                continue
            if args.only_labeled and not row.get("terrain_label"):
                continue
            yield row["tile_id"], Path(row["patch_path"])
    else:
        paths = sorted(p for p in args.hr_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        if not paths:
            raise SystemExit(f"no HR images found in {args.hr_dir}")
        for p in paths:
            yield p.stem, p


def process_geotiff(hr_src: Path, out_id: str, hr_out: Path, lr_out: Path,
                     cfg: dict, rng) -> dict:
    arr, profile, tags = pio.read_geotiff(hr_src)
    norm_mode = cfg["output"]["normalization"]
    normalized, norm_scale = pio.normalize_for_degradation(arr, norm_mode)

    result = degrade(normalized, cfg, rng)
    lr_native = pio.denormalize_lr(result["lr_image"], norm_scale, arr.dtype)

    hr_dst = hr_out / f"{out_id}.tif"
    lr_dst = lr_out / f"{out_id}.tif"

    # HR: lossless copy of the ground-truth patch (unchanged array + geo)
    pio.write_geotiff(hr_dst, arr, profile, profile["transform"], tags)
    # LR: degraded, with a geotransform scaled by the SR factor
    lr_transform = pio.scaled_transform(profile["transform"], result["params"]["scale_factor"])
    pio.write_geotiff(lr_dst, lr_native, profile, lr_transform, tags)

    params = dict(result["params"])
    params["normalization"] = norm_mode
    params["norm_scale"] = norm_scale
    return {
        "id": out_id, "hr_path": str(hr_dst), "lr_path": str(lr_dst),
        "format": "geotiff", "hr_shape": list(arr.shape),
        "lr_shape": list(lr_native.shape), "degradation_params": params,
    }


def process_png(hr_src: Path, out_id: str, hr_out: Path, lr_out: Path,
                 cfg: dict, rng) -> dict:
    hr = pio.read_png_gray_float(hr_src)
    result = degrade(hr, cfg, rng)
    hr_dst = hr_out / f"{out_id}.png"
    lr_dst = lr_out / f"{out_id}.png"
    pio.write_png_gray_float(hr, hr_dst)
    pio.write_png_gray_float(result["lr_image"], lr_dst)
    return {
        "id": out_id, "hr_path": str(hr_dst), "lr_path": str(lr_dst),
        "format": "png", "hr_shape": list(hr.shape),
        "lr_shape": list(result["lr_image"].shape),
        "degradation_params": result["params"],
    }


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path, help="stage 4 labeled manifest")
    src.add_argument("--hr-dir", type=Path, help="directory of HR images")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--config", default=Path("configs/degradation.yaml"), type=Path)
    ap.add_argument("--only-labeled", action="store_true",
                     help="with --manifest: skip patches that have no terrain_label")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_format = cfg.get("output", {}).get("format", "geotiff")
    rng = np.random.default_rng(cfg.get("seed"))

    hr_out = args.out_dir / "hr"
    lr_out = args.out_dir / "lr"
    hr_out.mkdir(parents=True, exist_ok=True)
    lr_out.mkdir(parents=True, exist_ok=True)

    process = process_geotiff if out_format == "geotiff" else process_png

    manifest = []
    for out_id, hr_src in collect_hr_items(args):
        row = process(hr_src, out_id, hr_out, lr_out, cfg, rng)
        manifest.append(row)
        print(f"  {out_id}: HR {row['hr_shape']} -> LR {row['lr_shape']} "
              f"({row['degradation_params']['downsample_method']}, {row['format']})")

    if not manifest:
        raise SystemExit("no patches processed — check --manifest filters or --hr-dir contents")

    manifest_path = args.out_dir / "degradation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest)} pairs ({out_format}) -> {args.out_dir}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
