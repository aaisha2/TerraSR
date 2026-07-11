"""Stage 6c: sanity report on the packaged, split dataset. Surfaces the
things that must be caught BEFORE training, not after (docs/plan §3):

  - per-terrain patch counts overall and per split
  - which terrains fall below the per-class floor (the water/mountain
    scarcity risk from the plan)
  - true PAN vs pseudo-PAN balance (pseudo-PAN is supplementary only —
    confirmed with supervisor — so it shouldn't dominate any terrain)
  - terrain leakage check: no source scene spanning multiple splits

Usage:
    python dataset_stats.py --manifest out/dataset/dataset_manifest_split.parquet
"""
import argparse
from pathlib import Path

import pandas as pd
import yaml


def load(manifest: Path) -> pd.DataFrame:
    if manifest.suffix == ".parquet":
        return pd.read_parquet(manifest)
    return pd.read_csv(manifest)


def print_table(title: str, series: pd.Series):
    print(f"\n{title}")
    for idx, val in series.items():
        print(f"  {str(idx):<20} {val}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                     help="split manifest from split_train_val_test.py")
    ap.add_argument("--config", default=Path("configs/split.yaml"), type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["stats"]
    floor = cfg["per_class_floor"]
    scarcity_watch = cfg.get("scarcity_watch", [])

    df = load(args.manifest)
    total = len(df)
    print(f"dataset: {total} patches, {df['source_scene'].nunique()} scenes, "
          f"{df['terrain_label'].nunique()} terrains")

    print_table("patches per terrain:", df["terrain_label"].value_counts())

    if "split" in df.columns:
        print("\npatches per terrain x split:")
        pivot = df.pivot_table(index="terrain_label", columns="split",
                                values="tile_id", aggfunc="count", fill_value=0)
        print(pivot.to_string())

    # per-class floor check
    print(f"\nper-class floor check (floor = {floor}):")
    counts = df["terrain_label"].value_counts().to_dict()
    for terrain, n in sorted(counts.items(), key=lambda kv: kv[1]):
        flag = "  <-- BELOW FLOOR" if n < floor else ""
        watch = "  (scarcity watch)" if terrain in scarcity_watch else ""
        print(f"  {terrain:<20} {n:>6}{flag}{watch}")
    missing_watch = [t for t in scarcity_watch if t not in counts]
    for terrain in missing_watch:
        print(f"  {terrain:<20} {0:>6}  <-- ABSENT (scarcity watch)")

    # true vs pseudo PAN
    if "pseudo_pan" in df.columns:
        print_table("PAN provenance:", df["pseudo_pan"].value_counts())

    # leakage guard: a scene must live in exactly one split
    if "split" in df.columns:
        leaked = (df.groupby("source_scene")["split"].nunique() > 1)
        n_leaked = int(leaked.sum())
        if n_leaked:
            print(f"\nERROR: {n_leaked} scene(s) span multiple splits — geographic "
                  f"block split is broken:")
            for scene in leaked[leaked].index:
                print(f"  {scene}")
        else:
            print("\nleakage check: OK -- every source scene is confined to one split")


if __name__ == "__main__":
    main()
