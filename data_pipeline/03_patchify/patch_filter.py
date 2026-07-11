"""Stage 3b: drop nodata/blank/saturated patches from a tile_extractor.py
manifest. Non-destructive by default — writes a filtered manifest with a
keep/drop flag and reasons per patch; pass --move-rejected to physically
relocate dropped patch files into a `rejected/` subfolder.

"Saturated" is a crude cloud/glare proxy (fraction of pixels near the
scene's own observed max), not a real cloud mask — good enough to catch
obviously blown-out patches without needing per-source cloud metadata.

Usage:
    python patch_filter.py --manifest out/patches/patch_manifest.json
    python patch_filter.py --manifest out/patches/patch_manifest.json --move-rejected
"""
import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import rasterio
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _common import patch_stats  # noqa: E402


def compute_scene_maxes(manifest: list) -> dict:
    """Approximate each source scene's max DN from the max observed across
    its own patches (avoids re-opening the original, possibly huge, scene)."""
    scene_max = defaultdict(float)
    for row in manifest:
        with rasterio.open(row["patch_path"]) as ds:
            scene_max[row["source_scene"]] = max(scene_max[row["source_scene"]], float(ds.read(1).max()))
    return scene_max


def evaluate_patch(patch_path: str, scene_max: float, filter_cfg: dict, nodata_value: int) -> dict:
    with rasterio.open(patch_path) as ds:
        arr = ds.read(1)
    stats = patch_stats(arr, nodata_value, filter_cfg["saturated_percentile"], scene_max)

    reasons = []
    if stats["nodata_fraction"] > filter_cfg["max_nodata_fraction"]:
        reasons.append(f"nodata_fraction={stats['nodata_fraction']:.3f} > "
                        f"{filter_cfg['max_nodata_fraction']}")
    if stats["std_dev"] < filter_cfg["min_std_dev"]:
        reasons.append(f"std_dev={stats['std_dev']:.2f} < {filter_cfg['min_std_dev']}")
    if stats["saturated_fraction"] > filter_cfg["max_saturated_fraction"]:
        reasons.append(f"saturated_fraction={stats['saturated_fraction']:.3f} > "
                        f"{filter_cfg['max_saturated_fraction']}")

    stats["keep"] = len(reasons) == 0
    stats["reject_reasons"] = reasons
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--config", default=Path("configs/patchify.yaml"), type=Path)
    ap.add_argument("--out-manifest", type=Path, default=None,
                     help="default: <manifest_dir>/patch_manifest_filtered.json")
    ap.add_argument("--nodata-value", type=int, default=0)
    ap.add_argument("--move-rejected", action="store_true",
                     help="physically move dropped patches into rejected/ next to the manifest")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["filter"]
    manifest = json.loads(args.manifest.read_text())

    print(f"computing per-scene max DN over {len(manifest)} patches...")
    scene_max = compute_scene_maxes(manifest)

    kept, dropped = 0, 0
    for row in manifest:
        stats = evaluate_patch(row["patch_path"], scene_max[row["source_scene"]], cfg, args.nodata_value)
        row.update(stats)
        if row["keep"]:
            kept += 1
        else:
            dropped += 1

    if args.move_rejected:
        rejected_dir = args.manifest.parent / "rejected"
        rejected_dir.mkdir(exist_ok=True)
        for row in manifest:
            if not row["keep"]:
                src = Path(row["patch_path"])
                dst = rejected_dir / src.name
                shutil.move(str(src), str(dst))
                row["patch_path"] = str(dst)

    out_manifest = args.out_manifest or args.manifest.parent / "patch_manifest_filtered.json"
    out_manifest.write_text(json.dumps(manifest, indent=2))

    print(f"kept {kept}, dropped {dropped} (of {len(manifest)})")
    for row in manifest:
        if not row["keep"]:
            print(f"  DROP {row['tile_id']}: {'; '.join(row['reject_reasons'])}")
    print(f"\nfiltered manifest -> {out_manifest}")


if __name__ == "__main__":
    main()
