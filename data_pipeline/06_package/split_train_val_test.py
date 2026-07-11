"""Stage 6b: assign each patch to train/val/test by GEOGRAPHIC BLOCK — whole
source scenes go to one split, never divided (docs/plan §3). Random per-patch
splitting would leak near-duplicate neighbours across splits and inflate
metrics, so the scene is the atomic unit here.

Assignment is a normalized-deficit greedy: scenes (largest first, ties broken
by a seeded shuffle) each go to the split with the largest *proportional*
remaining deficit (target - current)/target. Using the proportional deficit
rather than the absolute one keeps small splits (val/test) from being starved
by the large train target — with many scenes it converges to the configured
ratios, and with few scenes it still spreads scenes across splits instead of
dumping everything into train.

Usage:
    python split_train_val_test.py --manifest out/dataset/dataset_manifest.parquet
"""
import argparse
import random
from pathlib import Path

import pandas as pd
import yaml


def assign_scenes_to_splits(scene_sizes: dict, ratios: dict, seed: int) -> dict:
    total = sum(scene_sizes.values())
    targets = {split: total * frac for split, frac in ratios.items()}
    current = {split: 0 for split in ratios}

    rng = random.Random(seed)
    scenes = list(scene_sizes.items())
    rng.shuffle(scenes)                                  # seeded tie-break
    scenes.sort(key=lambda kv: kv[1], reverse=True)      # largest block first

    assignment = {}
    for scene, size in scenes:
        def deficit(split):
            t = targets[split]
            return (t - current[split]) / t if t > 0 else float("-inf")
        chosen = max(ratios, key=deficit)
        assignment[scene] = chosen
        current[chosen] += size
    return assignment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path,
                     help="unified manifest from build_manifest.py (.parquet or .csv)")
    ap.add_argument("--config", default=Path("configs/split.yaml"), type=Path)
    ap.add_argument("--out-dir", type=Path, default=None,
                     help="default: alongside the input manifest")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["split"]
    ratios = cfg["ratios"]
    group_col = cfg["group_by"]
    seed = cfg["seed"]

    if args.manifest.suffix == ".parquet":
        df = pd.read_parquet(args.manifest)
    else:
        df = pd.read_csv(args.manifest)

    scene_sizes = df.groupby(group_col).size().to_dict()
    n_scenes = len(scene_sizes)
    if n_scenes < len(ratios):
        print(f"WARNING: only {n_scenes} geographic block(s) ({group_col}) for "
              f"{len(ratios)} splits — some splits may be empty or far from target "
              f"ratios. This is expected on a tiny sample; the real dataset has "
              f"many scenes per terrain.")

    assignment = assign_scenes_to_splits(scene_sizes, ratios, seed)
    df["split"] = df[group_col].map(assignment)

    out_dir = args.out_dir or args.manifest.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    split_parquet = out_dir / "dataset_manifest_split.parquet"
    split_csv = out_dir / "dataset_manifest_split.csv"
    df.to_parquet(split_parquet, index=False)
    df.to_csv(split_csv, index=False)

    # convenience per-split CSVs for training/eval scripts to point at directly
    for split in ratios:
        sub = df[df["split"] == split]
        sub.to_csv(out_dir / f"{split}.csv", index=False)

    print("scene -> split assignment:")
    for scene, split in sorted(assignment.items(), key=lambda kv: (kv[1], kv[0])):
        print(f"  {split:<6} {scene}  ({scene_sizes[scene]} patches)")

    print("\npatches per split (actual vs target):")
    total = len(df)
    for split, frac in ratios.items():
        n = int((df["split"] == split).sum())
        print(f"  {split:<6} {n:>5}  ({100*n/total:5.1f}%  target {100*frac:.0f}%)")

    print(f"\n-> {split_parquet}")
    print(f"-> {split_csv}")


if __name__ == "__main__":
    main()
