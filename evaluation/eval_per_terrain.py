"""Stage 9b: per-terrain PSNR/SSIM breakdown — the key deliverable for the
project's thesis. Whether the terrain-conditioned model actually beats the
baselines shows up here, terrain by terrain (e.g. urban edges, forest
texture), not just in the overall average.

Compares any number of checkpoints side by side and, when exactly two are
given, prints the per-terrain delta so the terrain-aware gain (or loss) is
explicit.

Usage:
    python evaluation/eval_per_terrain.py \
        --test-csv data/dataset/test.csv \
        --checkpoints checkpoints/swinir_baseline/best.pth checkpoints/terrasr/best.pth
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation._eval_common import (load_model_from_checkpoint,  # noqa: E402
                                       make_test_loader, run_sr_over_loader)
from training.train_utils import get_device  # noqa: E402


def per_terrain_table(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("terrain").agg(
        n=("psnr", "size"), psnr=("psnr", "mean"), ssim=("ssim", "mean"))
    return agg.sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", required=True, type=Path)
    ap.add_argument("--checkpoints", nargs="+", required=True, type=Path)
    ap.add_argument("--with-bicubic", action="store_true")
    ap.add_argument("--bicubic-scale", type=int, default=2)
    ap.add_argument("--terrain-config", default="configs/terrain_classes.yaml")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-csv", type=Path, default=None)
    args = ap.parse_args()

    device = get_device()
    loader = make_test_loader(args.test_csv, args.terrain_config, args.batch_size)
    print(f"device: {device} | test patches: {len(loader.dataset)}\n")

    tables = {}   # label -> per-terrain DataFrame
    if args.with_bicubic:
        df = run_sr_over_loader(loader, device, bicubic_scale=args.bicubic_scale)
        tables["bicubic"] = per_terrain_table(df)

    for ckpt in args.checkpoints:
        model, name, is_terrain = load_model_from_checkpoint(ckpt, device)
        df = run_sr_over_loader(loader, device, model=model, is_terrain=is_terrain)
        tables[f"{name}:{ckpt.parent.name}"] = per_terrain_table(df)

    for label, table in tables.items():
        print(f"=== {label} ===")
        print(table.to_string(float_format=lambda v: f"{v:.3f}"))
        print()

    # explicit per-terrain delta when comparing exactly two models
    labels = list(tables)
    if len(labels) == 2:
        a, b = tables[labels[0]], tables[labels[1]]
        joined = a.join(b, lsuffix="_a", rsuffix="_b")
        joined["dPSNR"] = joined["psnr_b"] - joined["psnr_a"]
        joined["dSSIM"] = joined["ssim_b"] - joined["ssim_a"]
        print(f"=== delta ({labels[1]} - {labels[0]}) ===")
        print(joined[["dPSNR", "dSSIM"]].to_string(float_format=lambda v: f"{v:+.3f}"))
        print()

    if args.out_csv:
        combined = pd.concat({k: v for k, v in tables.items()}, names=["model"])
        combined.to_csv(args.out_csv)
        print(f"-> {args.out_csv}")


if __name__ == "__main__":
    main()
