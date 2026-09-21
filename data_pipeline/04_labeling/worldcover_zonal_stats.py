"""Stage 4a: compute ESA WorldCover class pixel-counts for every patch in a
stage 3 (filtered) manifest.

Streams only the small window each patch needs directly from the public
WorldCover COG tiles on S3 via GDAL's /vsicurl/ (verified 2026-07-11 --
no local WorldCover download or mosaic needed at all).

Resumable: progress is saved to the output manifest every SAVE_EVERY patches,
and a re-run reuses every histogram already computed (failed patches are
retried). Pass --fresh to recompute everything.

Usage:
    python worldcover_zonal_stats.py --manifest out/patches/patch_manifest_filtered.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _common import class_histogram, read_worldcover_window_for_patch  # noqa: E402

SAVE_EVERY = 500


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                     help="stage 3 patch manifest (filtered or unfiltered)")
    ap.add_argument("--config", default=Path("configs/terrain_classes.yaml"), type=Path)
    ap.add_argument("--out-manifest", type=Path, default=None,
                     help="default: <manifest_dir>/patch_manifest_zonal.json")
    ap.add_argument("--only-kept", action="store_true",
                     help="skip patches with keep=false (from patch_filter.py)")
    ap.add_argument("--fresh", action="store_true",
                     help="ignore histograms saved by a previous run and recompute all")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    manifest = json.loads(args.manifest.read_text())
    out_manifest = args.out_manifest or args.manifest.parent / "patch_manifest_zonal.json"

    # histograms already computed by an earlier (possibly interrupted) run
    previous = {}
    if out_manifest.exists() and not args.fresh:
        for r in json.loads(out_manifest.read_text()):
            if r.get("worldcover_histogram") is not None:
                previous[r["tile_id"]] = r

    # Apply ALL reusable results before computing anything, so every
    # intermediate save below still contains them (otherwise a second
    # interruption would drop results for rows not yet reached).
    todo, n_reused, n_skipped = [], 0, 0
    for row in manifest:
        if args.only_kept and row.get("keep") is False:
            n_skipped += 1
            continue
        prev = previous.get(row["tile_id"])
        if prev is not None:
            row["worldcover_histogram"] = prev["worldcover_histogram"]
            row["worldcover_pixel_count"] = prev.get("worldcover_pixel_count")
            n_reused += 1
        else:
            todo.append(row)
    if n_reused:
        print(f"resuming: {n_reused} patches already labelled, {len(todo)} to go")

    n_done = n_failed = n_since_save = 0
    for row in todo:
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
        n_since_save += 1
        if n_since_save >= SAVE_EVERY:
            atomic_write_json(out_manifest, manifest)
            n_since_save = 0
            print(f"  progress saved: {n_done + n_reused} labelled", flush=True)

    atomic_write_json(out_manifest, manifest)

    print(f"zonal stats: {n_done} computed, {n_reused} reused from previous run, "
          f"{n_skipped} skipped (keep=false), {n_failed} failed")
    print(f"manifest -> {out_manifest}")


if __name__ == "__main__":
    main()
