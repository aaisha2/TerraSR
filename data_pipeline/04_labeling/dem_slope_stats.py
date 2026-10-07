"""Stage 4a-bis: per-patch terrain SHAPE statistics (slope, local relief) from
the Copernicus DEM GLO-30, so the Mountain class is actually obtainable.

Why this exists: ESA WorldCover is a land-cover product with no landform
classes, so stage 4b could only assign Mountain through a hand-maintained
`scene_terrain_overrides` dict of scene-name substrings - and that dict was
empty. Mountain was in the terrain index (and so had an embedding row in the
stage 8 model and a weight in the terrain-aware loss) but could never be
assigned to any patch. Slope comes from a DEM, so that is where it now comes
from.

Resumable and cached exactly like worldcover_zonal_stats.py: DEM tiles are
downloaded once (~39 MB per 1-degree tile), progress is saved every
SAVE_EVERY patches, and a re-run reuses everything already computed. Tiles
the dataset does not publish (open ocean) are recorded as missing and not
re-requested.

Usage:
    python dem_slope_stats.py --manifest out/patches/patch_manifest_zonal.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _common import read_dem_stats_for_patch  # noqa: E402

SAVE_EVERY = 500
DEM_FIELDS = ("slope_mean_deg", "slope_p90_deg", "relief_m",
              "elevation_mean_m", "dem_pixels", "dem_tile", "valid")


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def copy_dem_fields(dst: dict, src: dict) -> None:
    dst["dem_stats"] = {k: src.get(k) for k in DEM_FIELDS if k in src}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                     help="stage 4a output (patch_manifest_zonal.json)")
    ap.add_argument("--config", default=Path("configs/terrain_classes.yaml"), type=Path)
    ap.add_argument("--out-manifest", type=Path, default=None,
                     help="default: overwrite the input manifest in place")
    ap.add_argument("--only-kept", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                     help="ignore DEM stats saved by a previous run and recompute")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    dem_cfg = (cfg.get("mountain") or {}).get("dem_source") or {}
    manifest = json.loads(args.manifest.read_text())
    out_manifest = args.out_manifest or args.manifest

    previous = {}
    if not args.fresh:
        for r in manifest:
            if r.get("dem_stats") is not None:
                previous[r["tile_id"]] = r["dem_stats"]

    # apply every reusable result before computing, so an interrupted run's
    # intermediate saves never drop results for rows not yet reached
    todo, n_reused, n_skipped = [], 0, 0
    for row in manifest:
        if args.only_kept and row.get("keep") is False:
            n_skipped += 1
            continue
        prev = previous.get(row["tile_id"])
        if prev is not None:
            row["dem_stats"] = prev
            n_reused += 1
        else:
            todo.append(row)
    if n_reused:
        print(f"resuming: {n_reused} patches already have DEM stats, {len(todo)} to go")

    n_done = n_nodem = n_failed = n_since_save = 0
    for row in todo:
        try:
            stats = read_dem_stats_for_patch(row["patch_path"], dem_cfg)
            copy_dem_fields(row, stats)
            if stats.get("valid"):
                n_done += 1
            else:
                n_nodem += 1
        except FileNotFoundError as e:
            # no DEM tile published here (ocean) - a normal outcome
            row["dem_stats"] = {"valid": False, "reason": str(e)}
            n_nodem += 1
        except Exception as e:
            print(f"  FAILED {row['tile_id']}: {e}")
            row["dem_stats"] = None
            n_failed += 1
        n_since_save += 1
        if n_since_save >= SAVE_EVERY:
            atomic_write_json(out_manifest, manifest)
            n_since_save = 0
            print(f"  progress saved: {n_done + n_reused} with DEM stats", flush=True)

    atomic_write_json(out_manifest, manifest)
    print(f"DEM slope stats: {n_done} computed, {n_reused} reused, "
          f"{n_nodem} without DEM coverage, {n_skipped} skipped (keep=false), "
          f"{n_failed} failed")
    print(f"manifest -> {out_manifest}")


if __name__ == "__main__":
    main()
