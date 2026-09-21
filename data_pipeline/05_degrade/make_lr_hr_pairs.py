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

Resumable: the degradation manifest is saved every SAVE_EVERY pairs, and a
re-run skips every pair whose HR and LR files already exist (outputs are
written atomically, so an existing file is always complete). Pass --fresh to
regenerate everything. When configs/degradation.yaml sets a `seed`, each pair
gets its own RNG derived from (seed, patch id), so a resumed run produces the
same pairs as an uninterrupted one.

Usage:
    python make_lr_hr_pairs.py --hr-dir tests/sample_images --out-dir out/pairs
    python make_lr_hr_pairs.py --manifest out/patches/patch_manifest_labeled.json --out-dir out/pairs
"""
import argparse
import json
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from degradation_pipeline import degrade  # noqa: E402
import patch_io as pio  # noqa: E402

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SAVE_EVERY = 500


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def pair_rng(seed, out_id: str, shared_rng):
    """Per-pair RNG when a seed is configured (resume-safe and order-
    independent); otherwise the shared unseeded generator."""
    if seed is None:
        return shared_rng
    return np.random.default_rng([int(seed), zlib.crc32(out_id.encode("utf-8"))])


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
    ap.add_argument("--fresh", action="store_true",
                     help="ignore pairs from a previous run and regenerate all")
    ap.add_argument("--verbose", action="store_true", help="print one line per pair")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    out_format = cfg.get("output", {}).get("format", "geotiff")
    seed = cfg.get("seed")
    shared_rng = np.random.default_rng(seed)

    hr_out = args.out_dir / "hr"
    lr_out = args.out_dir / "lr"
    hr_out.mkdir(parents=True, exist_ok=True)
    lr_out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "degradation_manifest.json"

    process = process_geotiff if out_format == "geotiff" else process_png

    # pairs completed by an earlier (possibly interrupted) run
    previous = {}
    if manifest_path.exists() and not args.fresh:
        for r in json.loads(manifest_path.read_text()):
            if Path(r["hr_path"]).exists() and Path(r["lr_path"]).exists():
                previous[r["id"]] = r

    items = list(collect_hr_items(args))
    if not items:
        raise SystemExit("no patches processed — check --manifest filters or --hr-dir contents")

    # slot every reused pair in first, so intermediate saves keep them all
    manifest, todo = [], []
    for out_id, hr_src in items:
        manifest.append(previous.get(out_id))
        if manifest[-1] is None:
            todo.append((len(manifest) - 1, out_id, hr_src))
    n_reused = len(items) - len(todo)
    if n_reused:
        print(f"resuming: {n_reused} pairs already done, {len(todo)} to go")

    for n, (slot, out_id, hr_src) in enumerate(todo, 1):
        row = process(hr_src, out_id, hr_out, lr_out, cfg, pair_rng(seed, out_id, shared_rng))
        manifest[slot] = row
        if args.verbose:
            print(f"  {out_id}: HR {row['hr_shape']} -> LR {row['lr_shape']} "
                  f"({row['degradation_params']['downsample_method']}, {row['format']})")
        if n % SAVE_EVERY == 0:
            atomic_write_json(manifest_path, [r for r in manifest if r is not None])
            print(f"  progress saved: {n_reused + n}/{len(items)} pairs", flush=True)

    atomic_write_json(manifest_path, manifest)
    print(f"\nwrote {len(todo)} new pairs, reused {n_reused} ({out_format}) -> {args.out_dir}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
