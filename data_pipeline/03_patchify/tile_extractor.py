"""Stage 3a: cut standardized scenes (stage 2 output) into a fixed-size patch
grid.

Each patch keeps the source scene's georeferencing (correct windowed
transform + CRS) and tags, plus new tags identifying where it came from
(needed downstream for the geographic train/val/test split in stage 6 —
patches from the same source scene must never be split across sets).

Resumable. A run can be interrupted at any point (Colab disconnect, power
loss, Ctrl-C) and re-running the same command picks up where it stopped:
  - each finished scene records its patch list in
    <out-dir>/_scene_manifests/<scene>.json, so finished scenes are skipped
    outright on the next run;
  - inside an unfinished scene, patches already on disk are not rewritten;
  - every file is written to a temp name and renamed when complete, so an
    interrupted write can never leave a truncated patch that a resume would
    mistake for a finished one.
Pass --fresh to ignore previous progress and rebuild everything.

Inline filtering (configs/patchify.yaml -> inline_filter: true) drops patches
that are mostly nodata or blank BEFORE writing them, instead of writing every
grid cell and discarding them in patch_filter.py. Scene collars can be half
of a satellite scene, so this avoids writing thousands of files that would be
deleted anyway. patch_filter.py still runs afterwards (it also applies the
scene-relative saturation check), and applies the same thresholds, so the
final kept set is unchanged.

Usage:
    python tile_extractor.py --in-scene standardized.tif --out-dir out/patches
    python tile_extractor.py --in-dir out/standardized --recursive --out-dir out/patches
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import rasterio
import yaml
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent))
from _common import grid_windows  # noqa: E402

SCENE_MANIFEST_DIR = "_scene_manifests"
NODATA_VALUE = 0          # stage 2 writes nodata=0 (STANDARD_NODATA)
PROGRESS_EVERY = 2000     # print a progress line every N grid cells


def atomic_write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, indent=2))
    os.replace(tmp, path)


def settings_from_cfg(cfg: dict) -> dict:
    """The parameters that determine a scene's patch set. A saved scene
    manifest is only reused if these match the current config."""
    s = {"patch_size": cfg["patch_size"], "stride": cfg["stride"],
         "drop_partial_edge": cfg["drop_partial_edge"],
         "inline_filter": bool(cfg.get("inline_filter", False))}
    if s["inline_filter"]:
        s["max_nodata_fraction"] = cfg["filter"]["max_nodata_fraction"]
        s["min_std_dev"] = cfg["filter"]["min_std_dev"]
    return s


def keep_patch(data: np.ndarray, settings: dict) -> bool:
    if not settings["inline_filter"]:
        return True
    if float(np.mean(data == NODATA_VALUE)) > settings["max_nodata_fraction"]:
        return False
    return float(np.std(data)) >= settings["min_std_dev"]


def extract_patches(scene_path: Path, out_dir: Path, settings: dict, fresh: bool):
    """Returns (rows, status) where status is 'reused' or a summary string."""
    source_stem = scene_path.stem
    manifest_file = out_dir / SCENE_MANIFEST_DIR / f"{source_stem}.json"

    if manifest_file.exists() and not fresh:
        saved = json.loads(manifest_file.read_text())
        if saved.get("settings") != settings:
            raise SystemExit(
                f"{scene_path.name}: patchify settings changed since this scene was "
                f"processed ({saved.get('settings')} -> {settings}). "
                f"Re-run with --fresh to rebuild the patches.")
        return saved["rows"], "reused (already done)"

    ps = settings["patch_size"]
    rows, n_written, n_existing, n_dropped = [], 0, 0, 0
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(scene_path) as src:
        source_tags = src.tags()
        profile = src.profile.copy()
        profile.update(height=ps, width=ps, tiled=True, blockxsize=256, blockysize=256)
        profile.pop("BIGTIFF", None)

        cells = list(grid_windows(src.height, src.width, ps, settings["stride"],
                                  settings["drop_partial_edge"]))
        for i, (row, col, row_off, col_off) in enumerate(cells, 1):
            tile_id = f"{source_stem}_r{row:04d}_c{col:04d}"
            patch_path = out_dir / f"{tile_id}.tif"
            record = {"tile_id": tile_id, "patch_path": str(patch_path),
                      "source_scene": scene_path.name, "row": row, "col": col,
                      "row_off": row_off, "col_off": col_off}

            if patch_path.exists() and not fresh:
                rows.append(record)
                n_existing += 1
            else:
                window = Window(col_off, row_off, ps, ps)
                data = src.read(1, window=window)
                if not keep_patch(data, settings):
                    n_dropped += 1
                else:
                    profile["transform"] = src.window_transform(window)
                    tmp = patch_path.with_name(patch_path.stem + ".partial.tif")
                    with rasterio.open(tmp, "w", **profile) as dst:
                        dst.write(data, 1)
                        dst.update_tags(**source_tags, source_scene=scene_path.name,
                                        tile_id=tile_id, patch_row=str(row),
                                        patch_col=str(col))
                    os.replace(tmp, patch_path)
                    rows.append(record)
                    n_written += 1

            if i % PROGRESS_EVERY == 0:
                print(f"    {scene_path.name}: {i}/{len(cells)} cells", flush=True)

    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(manifest_file, {"settings": settings, "scene": scene_path.name,
                                      "n_grid_cells": len(cells), "n_dropped_inline": n_dropped,
                                      "rows": rows})
    status = f"{n_written} written, {n_existing} already on disk, {n_dropped} dropped inline"
    return rows, status


def find_scenes(in_dir: Path, recursive: bool):
    globber = in_dir.rglob if recursive else in_dir.glob
    found = set(globber("*.tif")) | set(globber("*.tiff"))
    # skip temp files left by an interrupted stage-2 write
    return sorted(p for p in found if ".partial" not in p.name)


def main():
    ap = argparse.ArgumentParser()
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--in-scene", type=Path, help="a single standardized GeoTIFF")
    src_group.add_argument("--in-dir", type=Path, help="a directory of standardized GeoTIFFs (*.tif)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--config", default=Path("configs/patchify.yaml"), type=Path)
    ap.add_argument("--recursive", action="store_true",
                     help="with --in-dir: also search sub-directories (per-source folders)")
    ap.add_argument("--fresh", action="store_true",
                     help="ignore previous progress and rebuild every patch")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    settings = settings_from_cfg(cfg)

    scenes = [args.in_scene] if args.in_scene else find_scenes(args.in_dir, args.recursive)
    if not scenes:
        raise SystemExit(f"no scenes found in {args.in_dir}")
    print(f"{len(scenes)} scene(s); settings: {settings}")

    all_rows = []
    for n, scene_path in enumerate(scenes, 1):
        t0 = time.time()
        rows, status = extract_patches(scene_path, args.out_dir, settings, args.fresh)
        all_rows.extend(rows)
        print(f"[{n}/{len(scenes)}] {scene_path.name}: {len(rows)} patches  "
              f"({status}, {time.time() - t0:.0f}s)", flush=True)

    manifest_path = args.out_dir / "patch_manifest.json"
    atomic_write_json(manifest_path, all_rows)
    print(f"\n{len(all_rows)} patches total -> {args.out_dir}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
