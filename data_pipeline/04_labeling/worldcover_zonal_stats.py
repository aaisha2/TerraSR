"""Stage 4a: compute ESA WorldCover class pixel-counts for every patch in a
stage 3 (filtered) manifest.

Streams only the small window each patch needs directly from the public
WorldCover COG tiles on S3 via GDAL's /vsicurl/ (verified 2026-07-11 --
no local WorldCover download or mosaic needed at all).

Usage:
    python worldcover_zonal_stats.py --manifest out/patches/patch_manifest_filtered.json
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _common import class_histogram, read_worldcover_window_for_patch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                     help="stage 3 patch manifest (filtered or unfiltered)")
    ap.add_argument("--config", default=Path("configs/terrain_classes.yaml"), type=Path)
    ap.add_argument("--out-manifest", type=Path, default=None,
                     help="default: <manifest_dir>/patch_manifest_zonal.json")
    ap.add_argument("--only-kept", action="store_true",
                     help="skip patches with keep=false (from patch_filter.py)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())

    n_done, n_skipped, n_failed = 0, 0, 0
    for row in manifest:
        if args.only_kept and row.get("keep") is False:
            n_skipped += 1
            continue
        try:
            wc_arr = read_worldcover_window_for_patch(row["patch_path"], cfg)
        except Exception as e:
            print(f"  FAILED {row['tile_id']}: {e}")
            row["worldcover_histogram"] = None
            n_failed += 1
            continue

        row["worldcover_histogram"] = class_histogram(wc_arr)
        row["worldcover_pixel_count"] = int(wc_arr.size)
        n_done += 1

    out_manifest = args.out_manifest or args.manifest.parent / "patch_manifest_zonal.json"
    out_manifest.write_text(json.dumps(manifest, indent=2))

    print(f"zonal stats: {n_done} done, {n_skipped} skipped (keep=false), {n_failed} failed")
    print(f"manifest -> {out_manifest}")


if __name__ == "__main__":
    main()
