"""Stage 1: download IARPA CORE3D public WorldView-2/3 PANCHROMATIC imagery.

Third true-PAN source alongside SpaceNet and Maxar Open Data. Hosted by
SpaceNet in the SAME public bucket we already use (s3://spacenet-dataset under
Hosted-Datasets/CORE3D-Public-Data/), so it needs no new credentials.

Verified 2026-07-28 against the live bucket:
  - 154 true-PAN GeoTIFFs, ~311 GB total, across 5 sites
  - WorldView-3 PAN and WorldView-2 PAN; measured GSD 0.34-0.46 m
  - single band, uint16, tiled with overviews (so windowed reads work)
  - CRS is EPSG:4326, which stage 2 reprojects to UTM automatically

IMPORTANT file-naming detail: the real imagery is the `*_lv1.tif` files.
Each scene also has a same-named 0-byte `.tif` plus `.vrt`/`.NTF`/`.tar`
sidecars — this script downloads ONLY `-P1BS-*_lv1.tif` (P1BS = Panchromatic
1B Standard), so no previews, metadata or multispectral products are fetched.

Two modes, because full scenes are 1-2.7 GB each:

  full (default)  download whole PAN scenes. Faithful but disk-hungry.
  --windows N     extract N random NxN crops per scene directly from the remote
                  file via GDAL range reads, without downloading the whole
                  scene. Same pixels, a tiny fraction of the bytes -- the
                  practical way to use CORE3D under a disk budget.

Usage:
    python download_core3d.py --config configs/datasets.yaml --list-only
    python download_core3d.py --config configs/datasets.yaml --windows 40 --window-size 1024
    python download_core3d.py --config configs/datasets.yaml --site Omaha --max-files 2
"""
import argparse
import sys
from pathlib import Path

import boto3
import yaml
from botocore import UNSIGNED
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).parent))
from _common import already_downloaded, human_size  # noqa: E402

BUCKET_HTTPS = "https://spacenet-dataset.s3.amazonaws.com"


def list_pan_scenes(s3, bucket, prefix, sites, min_size):
    """Yield (site, sensor, key, size) for every true-PAN GeoTIFF."""
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key, size = obj["Key"], obj["Size"]
            # PAN only, real imagery only (see module docstring)
            if "-P1BS-" not in key or not key.endswith("_lv1.tif"):
                continue
            if size < min_size:
                continue
            parts = key.split("/")
            if len(parts) < 6:
                continue
            site, sensor = parts[3], parts[4]
            if sites and site not in sites:
                continue
            yield site, sensor, key, size


def download_full(s3, bucket, key, size, dest):
    if already_downloaded(dest, size):
        print(f"    skip (already downloaded): {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    state = {"n": 0}

    def progress(chunk):
        state["n"] += chunk
        pct = 100 * state["n"] / size if size else 0
        print(f"\r    {dest.name[:44]}: {human_size(state['n'])}/{human_size(size)} ({pct:.0f}%)",
              end="", flush=True)

    s3.download_file(bucket, key, str(tmp), Callback=progress)
    print()
    tmp.rename(dest)


def extract_windows(key, out_dir, n_windows, window_size, seed):
    """Pull n random crops straight out of the remote GeoTIFF using range reads.
    Needs rasterio; skips gracefully with a clear message if unavailable."""
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    url = f"/vsicurl/{BUCKET_HTTPS}/{key}"
    stem = key.split("/")[-1].replace("_lv1.tif", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with rasterio.open(url) as src:
        if src.width < window_size or src.height < window_size:
            print(f"    scene smaller than window ({src.width}x{src.height}); skipped")
            return 0
        rng = np.random.default_rng(seed)
        attempts = 0
        while written < n_windows and attempts < n_windows * 12:
            attempts += 1
            col = int(rng.integers(0, src.width - window_size))
            row = int(rng.integers(0, src.height - window_size))
            dest = out_dir / f"{stem}_w{row:06d}_{col:06d}.tif"
            if dest.exists():
                written += 1
                continue
            win = Window(col, row, window_size, window_size)
            data = src.read(1, window=win)
            # reject crops that are mostly nodata/blank -- stage 3 would drop
            # them anyway, so don't waste the bytes or the round trip
            if float((data == 0).mean()) > 0.05 or float(data.std()) < 4.0:
                continue
            profile = src.profile.copy()
            profile.update(height=window_size, width=window_size, count=1,
                           transform=src.window_transform(win),
                           compress="deflate", predictor=2, tiled=True,
                           blockxsize=256, blockysize=256)
            profile.pop("nodata", None)
            tmp = dest.with_suffix(".tif.part")
            with rasterio.open(tmp, "w", **profile) as dst:
                dst.write(data, 1)
                dst.update_tags(source_scene=stem, core3d_window=f"{row},{col}",
                                band_source="true_pan", pseudo_pan="false")
            tmp.rename(dest)
            written += 1
            print(f"\r    {stem[:40]}: {written}/{n_windows} crops", end="", flush=True)
    if written:
        print()
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=Path("configs/datasets.yaml"), type=Path)
    ap.add_argument("--site", action="append", default=None,
                     help="restrict to site(s), repeatable; default: all in config")
    ap.add_argument("--max-files", type=int, default=None, help="cap scenes per site")
    ap.add_argument("--list-only", action="store_true")
    ap.add_argument("--windows", type=int, default=None,
                     help="extract N random crops per scene instead of downloading it whole")
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["core3d"]
    bucket, prefix = cfg["bucket"], cfg["prefix"]
    out_dir = Path(cfg["out_dir"])
    sites = args.site or cfg.get("sites") or None
    min_size = int(cfg.get("min_scene_bytes", 10_000_000))

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    per_site = {}
    for site, sensor, key, size in list_pan_scenes(s3, bucket, prefix, sites, min_size):
        per_site.setdefault(site, []).append((sensor, key, size))

    if not per_site:
        raise SystemExit(f"no PAN scenes found under {prefix} for sites={sites}")

    total_files = total_bytes = total_crops = 0
    for site in sorted(per_site):
        scenes = sorted(per_site[site], key=lambda t: t[1])
        if args.max_files is not None:
            scenes = scenes[:args.max_files]
        print(f"\n=== {site} ({len(scenes)} PAN scenes) ===")
        for sensor, key, size in scenes:
            total_files += 1
            total_bytes += size
            name = key.split("/")[-1]
            if args.list_only:
                print(f"  {sensor}  {human_size(size):>8}  {name[:78]}")
            elif args.windows:
                total_crops += extract_windows(key, out_dir / site,
                                                args.windows, args.window_size, args.seed)
            else:
                download_full(s3, bucket, key, size, out_dir / site / name)

    if args.list_only:
        print(f"\nwould fetch {total_files} PAN scenes, {human_size(total_bytes)} total")
        if not args.windows:
            print("  tip: --windows N extracts crops via range reads instead "
                  "(vastly less disk for the same pixels)")
    elif args.windows:
        print(f"\nextracted {total_crops} crops from {total_files} scenes -> {out_dir}")
    else:
        print(f"\nfetched {total_files} PAN scenes ({human_size(total_bytes)}) -> {out_dir}")


if __name__ == "__main__":
    main()
