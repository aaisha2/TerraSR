"""Stage 3a: cut a standardized scene (stage 2 output) into a fixed-size
patch grid. No filtering here — every complete grid cell is written; run
patch_filter.py afterwards to drop nodata/blank/saturated patches.

Each patch keeps the source scene's georeferencing (correct windowed
transform + CRS) and tags, plus new tags identifying where it came from
(needed downstream for the geographic train/val/test split in stage 6 —
patches from the same source scene must never be split across sets).

Usage:
    python tile_extractor.py --in-scene standardized.tif --out-dir out/patches
    python tile_extractor.py --in-dir out/standardized_scenes --out-dir out/patches
"""
import argparse
import json
import sys
from pathlib import Path

import rasterio
import yaml
from rasterio.windows import Window

sys.path.insert(0, str(Path(__file__).parent))
from _common import grid_windows  # noqa: E402


def extract_patches(scene_path: Path, out_dir: Path, patch_size: int, stride: int,
                     drop_partial_edge: bool) -> list:
    source_stem = scene_path.stem
    manifest_rows = []

    with rasterio.open(scene_path) as src:
        source_tags = src.tags()
        for row, col, row_off, col_off in grid_windows(
                src.height, src.width, patch_size, stride, drop_partial_edge):
            window = Window(col_off, row_off, patch_size, patch_size)
            data = src.read(1, window=window)
            transform = src.window_transform(window)

            tile_id = f"{source_stem}_r{row:04d}_c{col:04d}"
            patch_path = out_dir / f"{tile_id}.tif"

            profile = src.profile.copy()
            profile.update(height=patch_size, width=patch_size, transform=transform)

            out_dir.mkdir(parents=True, exist_ok=True)
            with rasterio.open(patch_path, "w", **profile) as dst:
                dst.write(data, 1)
                dst.update_tags(**source_tags, source_scene=scene_path.name,
                                 tile_id=tile_id, patch_row=str(row), patch_col=str(col))

            manifest_rows.append({
                "tile_id": tile_id,
                "patch_path": str(patch_path),
                "source_scene": scene_path.name,
                "row": row,
                "col": col,
                "row_off": row_off,
                "col_off": col_off,
            })

    return manifest_rows


def main():
    ap = argparse.ArgumentParser()
    src_group = ap.add_mutually_exclusive_group(required=True)
    src_group.add_argument("--in-scene", type=Path, help="a single standardized GeoTIFF")
    src_group.add_argument("--in-dir", type=Path, help="a directory of standardized GeoTIFFs (*.tif)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--config", default=Path("configs/patchify.yaml"), type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    patch_size = cfg["patch_size"]
    stride = cfg["stride"]
    drop_partial_edge = cfg["drop_partial_edge"]

    scenes = [args.in_scene] if args.in_scene else sorted(args.in_dir.glob("*.tif"))
    if not scenes:
        raise SystemExit(f"no scenes found in {args.in_dir}")

    all_rows = []
    for scene_path in scenes:
        rows = extract_patches(scene_path, args.out_dir, patch_size, stride, drop_partial_edge)
        all_rows.extend(rows)
        print(f"{scene_path.name}: {len(rows)} patches")

    manifest_path = args.out_dir / "patch_manifest.json"
    manifest_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\n{len(all_rows)} patches total -> {args.out_dir}")
    print(f"manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
