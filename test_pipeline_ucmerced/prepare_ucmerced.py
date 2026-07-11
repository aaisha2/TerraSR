"""UC Merced test harness — stages 2-4 equivalent, adapted to UC Merced.

For each UC Merced tile it does, in one pass, what stages 2 (standardize),
3 (patchify) and 4 (label) do for the real pipeline — but adapted to UC
Merced's specifics:

  - stage 2: RGB -> pseudo-PAN grayscale (BT.709 luma, same weights as the
    real extract_pan_band.py), cast uint8 -> uint16 (same rule as the real
    cast_to_standard_dtype), written as a single-band GeoTIFF tagged
    pseudo_pan=true. A synthetic UTM CRS + 0.3 m transform is attached so the
    downstream GeoTIFF code (stage 5's georeferenced LR, etc.) runs identically.
  - stage 3: UC Merced tiles are already 256x256 = one patch each, so no
    tiling is needed; each image becomes one patch.
  - stage 4: terrain comes from the UC Merced land-use CLASS (folder name) via
    ucmerced_terrain.yaml, not ESA WorldCover (UC Merced tiles aren't
    georeferenced). Each image is its own source_scene, so stage 6's
    geographic-block split becomes a clean per-image split (no leakage).

Output is a stage-4-compatible labeled manifest, so the real stages 5 and 6
consume it unchanged.

    python test_pipeline_ucmerced/prepare_ucmerced.py \
        --raw-dir test_pipeline_ucmerced/data/raw \
        --out-standardized test_pipeline_ucmerced/data/standardized \
        --out-manifest test_pipeline_ucmerced/data/patches/patch_manifest_labeled.json \
        --config test_pipeline_ucmerced/configs/ucmerced_terrain.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
import yaml
from PIL import Image
from rasterio.transform import from_origin

LUMA_WEIGHTS = (0.2126, 0.7152, 0.0722)   # BT.709, same as extract_pan_band.py
SYNTH_CRS = "EPSG:32611"                   # arbitrary UTM zone (UC Merced ~ US)
SYNTH_PIXEL_M = 0.3                        # UC Merced ~1ft GSD


def rgb_to_pseudo_pan_uint16(rgb_u8: np.ndarray) -> np.ndarray:
    lum = sum(w * rgb_u8[..., i].astype(np.float64) for i, w in enumerate(LUMA_WEIGHTS))
    lum_u8 = np.clip(lum, 0, 255).astype(np.uint16)
    return (lum_u8 * 257).astype(np.uint16)   # uint8 -> uint16 (same as cast rule)


def write_patch(arr_u16: np.ndarray, out_path: Path, tile_id: str, terrain: str):
    h, w = arr_u16.shape
    transform = from_origin(0, 0, SYNTH_PIXEL_M, SYNTH_PIXEL_M)
    profile = {
        "driver": "GTiff", "height": h, "width": w, "count": 1, "dtype": "uint16",
        "crs": SYNTH_CRS, "transform": transform, "nodata": 0,
        "compress": "deflate", "predictor": 2, "tiled": True,
        "blockxsize": 256, "blockysize": 256,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr_u16, 1)
        dst.update_tags(pseudo_pan="true", band_source="rgb_luminance_bt709",
                        standardized="true", source="ucmerced",
                        terrain_label=terrain, tile_id=tile_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, type=Path)
    ap.add_argument("--out-standardized", required=True, type=Path)
    ap.add_argument("--out-manifest", required=True, type=Path)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--limit-per-class", type=int, default=None,
                     help="cap images per class (quick tests)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    class_to_terrain = cfg["ucmerced_class_to_terrain"]

    images_root = args.raw_dir / "UCMerced_LandUse" / "Images"
    if not images_root.exists():
        raise SystemExit(f"UC Merced images not found at {images_root} — run download_ucmerced.py first")

    manifest = []
    per_class_count = {}
    unmapped_classes = set()
    for img_path in sorted(images_root.rglob("*.tif")):
        cls = img_path.parent.name.lower()
        terrain = class_to_terrain.get(cls)
        if terrain is None:
            unmapped_classes.add(cls)
            continue
        if args.limit_per_class is not None:
            per_class_count[cls] = per_class_count.get(cls, 0) + 1
            if per_class_count[cls] > args.limit_per_class:
                continue

        tile_id = f"{cls}__{img_path.stem}"
        rgb = np.asarray(Image.open(img_path).convert("RGB"), dtype=np.uint8)
        if rgb.shape[0] < 8 or rgb.shape[1] < 8:
            continue
        pan = rgb_to_pseudo_pan_uint16(rgb)

        patch_path = args.out_standardized / f"{tile_id}.tif"
        write_patch(pan, patch_path, tile_id, terrain)

        manifest.append({
            "tile_id": tile_id,
            "patch_path": str(patch_path),
            "source_scene": tile_id,          # each image is its own scene
            "keep": True,
            "terrain_label": terrain,
            "terrain_purity": 1.0,
            "terrain_source": "ucmerced_class",
            "row": 0, "col": 0,
        })

    if not manifest:
        raise SystemExit("no images prepared — check --raw-dir and the class map")

    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_manifest.write_text(json.dumps(manifest, indent=2))

    from collections import Counter
    counts = Counter(r["terrain_label"] for r in manifest)
    print(f"prepared {len(manifest)} patches -> {args.out_standardized}")
    for terrain, n in counts.most_common():
        print(f"  {terrain:<20} {n}")
    if unmapped_classes:
        print(f"  (unmapped classes skipped: {sorted(unmapped_classes)})")
    print(f"labeled manifest -> {args.out_manifest}")


if __name__ == "__main__":
    main()
