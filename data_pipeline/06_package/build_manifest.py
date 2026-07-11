"""Stage 6a: join the stage 4 labeled patch manifest with the stage 5
LR/HR degradation manifest into one unified per-patch manifest — the single
source of truth every training/eval script reads.

One row per kept, labeled patch that also has an LR/HR pair:
    tile_id, hr_path, lr_path, terrain_label, terrain_purity, terrain_source,
    source_scene, pseudo_pan, downsample_method, ...

The join key is tile_id (labeled manifest) == id (degradation manifest);
stage 5 keys its outputs by the HR patch filename stem, which is the tile_id.

Note on HR/LR formats: stage 5 currently emits 8-bit PNG pairs (a carry-over
from its early synthetic smoke test). For the real training run stage 5
should be re-run in a mode that preserves the 16-bit GeoTIFF depth of the
PAN patches — the manifest schema here is unchanged by that (it only stores
paths), so this stage needs no edits when that upgrade lands.

Usage:
    python build_manifest.py \
        --labeled out/patches/patch_manifest_labeled.json \
        --degradation out/pairs/degradation_manifest.json \
        --out-dir out/dataset
"""
import argparse
import json
from pathlib import Path

import pandas as pd


CARRY_FROM_LABELED = [
    "tile_id", "source_scene", "terrain_label", "terrain_purity",
    "terrain_source", "row", "col",
]


def load_pseudo_pan_flag(patch_path: str) -> str:
    """Read the pseudo_pan tag off the HR GeoTIFF patch so true vs pseudo PAN
    is queryable straight from the manifest (needed for the primary-vs-
    supplementary training split and any ablation)."""
    try:
        import rasterio
        with rasterio.open(patch_path) as ds:
            return ds.tags().get("pseudo_pan", "unknown")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", required=True, type=Path,
                     help="stage 4 output (patch_manifest_labeled.json)")
    ap.add_argument("--degradation", required=True, type=Path,
                     help="stage 5 output (degradation_manifest.json)")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--skip-pseudo-pan-tag", action="store_true",
                     help="don't open each HR patch to read its pseudo_pan tag (faster)")
    args = ap.parse_args()

    labeled = json.loads(args.labeled.read_text())
    degradation = json.loads(args.degradation.read_text())

    lr_by_id = {row["id"]: row for row in degradation}

    rows = []
    n_no_pair, n_not_kept, n_no_label = 0, 0, 0
    for lab in labeled:
        if lab.get("keep") is False:
            n_not_kept += 1
            continue
        if not lab.get("terrain_label"):
            n_no_label += 1
            continue
        deg = lr_by_id.get(lab["tile_id"])
        if deg is None:
            n_no_pair += 1
            continue

        row = {k: lab.get(k) for k in CARRY_FROM_LABELED}
        row["hr_path"] = deg["hr_path"]
        row["lr_path"] = deg["lr_path"]
        row["downsample_method"] = deg.get("degradation_params", {}).get("downsample_method")
        row["scale_factor"] = deg.get("degradation_params", {}).get("scale_factor")
        row["pseudo_pan"] = ("unknown" if args.skip_pseudo_pan_tag
                              else load_pseudo_pan_flag(lab["patch_path"]))
        rows.append(row)

    if not rows:
        raise SystemExit("no patches survived the join — check that the labeled and "
                          "degradation manifests refer to the same patches")

    df = pd.DataFrame(rows).sort_values("tile_id").reset_index(drop=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.out_dir / "dataset_manifest.csv"
    parquet_path = args.out_dir / "dataset_manifest.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    print(f"unified manifest: {len(df)} patches")
    print(f"  skipped: {n_not_kept} not-kept, {n_no_label} unlabeled, {n_no_pair} without an LR/HR pair")
    print(f"  terrains: {df['terrain_label'].value_counts().to_dict()}")
    print(f"  scenes:   {df['source_scene'].nunique()}")
    print(f"-> {csv_path}")
    print(f"-> {parquet_path}")


if __name__ == "__main__":
    main()
