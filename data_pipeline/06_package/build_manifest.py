"""Stage 6a: join the stage 4 labeled patch manifest with the stage 5
LR/HR degradation manifest into one unified per-patch manifest — the single
source of truth every training/eval script reads.

One row per kept, labeled patch that also has an LR/HR pair:
    tile_id, hr_path, lr_path, terrain_label, terrain_purity, terrain_source,
    source_scene, pseudo_pan, downsample_method, ...

The join key is tile_id (labeled manifest) == id (degradation manifest);
stage 5 keys its outputs by the HR patch filename stem, which is the tile_id.

UNLABELLED PATCHES ARE KEPT BY DEFAULT (changed 2026-10-07). This stage used
to drop every patch without a terrain label. That was harmless while stage 4
had no purity threshold and labelled essentially everything, but stage 4 now
abstains on genuinely mixed patches — so the old behaviour would silently
delete a quarter of the dataset and bias what remains towards pure, easy
single-terrain scenes. Super-resolution does not need a terrain label; only
the conditioning does, and the model already has an "unknown" embedding row
for exactly this. Unlabelled patches therefore stay in the manifest with an
empty terrain_label, and `--drop-unlabeled` restores the old behaviour for a
labelled-only ablation.

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
    # label-quality fields (stage 4b) — carried through so training/eval can
    # filter on label confidence and the results report can state it
    "terrain_margin", "terrain_runner_up", "terrain_decisive",
    "terrain_unmapped_fraction", "terrain_landcover_under_mountain",
    # geography (stage 3) — required by the spatial-block split in stage 6b
    "center_lon", "center_lat", "bounds_wgs84",
]

# degradation parameters worth having as manifest columns, so a reader can
# audit what degradation each training pair actually received
CARRY_FROM_DEGRADATION = [
    "downsample_method", "scale_factor", "blur_kernel",
    "effective_nyquist_mtf", "noise_snr_at_full_signal", "jpeg_quality",
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
    ap.add_argument("--drop-unlabeled", action="store_true",
                     help="exclude patches with no terrain label (default: keep them; "
                          "they train through the model's unknown-terrain embedding)")
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
            if args.drop_unlabeled:
                continue
        deg = lr_by_id.get(lab["tile_id"])
        if deg is None:
            n_no_pair += 1
            continue

        row = {k: lab.get(k) for k in CARRY_FROM_LABELED}
        row["hr_path"] = deg["hr_path"]
        row["lr_path"] = deg["lr_path"]
        deg_params = deg.get("degradation_params", {})
        for key in CARRY_FROM_DEGRADATION:
            row[key] = deg_params.get(key)
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

    n_unlabeled_kept = int(df["terrain_label"].isna().sum()) if "terrain_label" in df else 0
    print(f"unified manifest: {len(df)} patches")
    print(f"  skipped: {n_not_kept} not-kept, {n_no_pair} without an LR/HR pair"
          + (f", {n_no_label} unlabeled (--drop-unlabeled)" if args.drop_unlabeled
             else ""))
    print(f"  terrains: {df['terrain_label'].value_counts().to_dict()}")
    if n_unlabeled_kept:
        print(f"  unlabeled kept: {n_unlabeled_kept} "
              f"({100*n_unlabeled_kept/len(df):.1f}%) - these train through the "
              f"model's unknown-terrain embedding")
    print(f"  scenes:   {df['source_scene'].nunique()}")
    if "center_lon" in df.columns and df["center_lon"].isna().all():
        print("  WARNING: no patch geography (center_lon/center_lat) in this manifest. "
              "The stage 6b spatial-block split needs it to prevent same-location "
              "leakage; run data_pipeline/06_package/backfill_patch_geo.py first.")
    print(f"-> {csv_path}")
    print(f"-> {parquet_path}")


if __name__ == "__main__":
    main()
