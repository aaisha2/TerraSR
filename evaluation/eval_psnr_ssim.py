"""Stage 9a: overall PSNR/SSIM on the test split for one or more checkpoints,
plus the bicubic interpolation floor. This is the top-line benchmarking table
the proposal requires (transformer vs CNN/GAN baselines).

Usage:
    python evaluation/eval_psnr_ssim.py \
        --test-csv data/dataset/test.csv \
        --checkpoints checkpoints/swinir_baseline/best.pth checkpoints/terrasr/best.pth \
        --with-bicubic
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation._eval_common import (load_model_from_checkpoint,  # noqa: E402
                                       make_test_loader, run_sr_over_loader)
from training.train_utils import get_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", required=True, type=Path)
    ap.add_argument("--checkpoints", nargs="*", default=[], type=Path)
    ap.add_argument("--with-bicubic", action="store_true", help="include the bicubic floor")
    ap.add_argument("--bicubic-scale", type=int, default=2)
    ap.add_argument("--terrain-config", default="configs/terrain_classes.yaml")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out-csv", type=Path, default=None,
                     help="optional: write the summary table to CSV")
    args = ap.parse_args()

    device = get_device()
    loader = make_test_loader(args.test_csv, args.terrain_config, args.batch_size)
    n = len(loader.dataset)
    print(f"device: {device} | test patches: {n}\n")

    rows = []
    if args.with_bicubic:
        df = run_sr_over_loader(loader, device, bicubic_scale=args.bicubic_scale)
        rows.append(("bicubic", df["psnr"].mean(), df["ssim"].mean()))

    for ckpt in args.checkpoints:
        model, name, is_terrain = load_model_from_checkpoint(ckpt, device)
        df = run_sr_over_loader(loader, device, model=model, is_terrain=is_terrain)
        rows.append((f"{name} ({ckpt.parent.name})", df["psnr"].mean(), df["ssim"].mean()))

    print(f"{'model':<34}{'PSNR (dB)':>12}{'SSIM':>10}")
    print("-" * 56)
    for name, psnr, ssim in rows:
        print(f"{name:<34}{psnr:>12.3f}{ssim:>10.4f}")

    if args.out_csv:
        import pandas as pd
        pd.DataFrame(rows, columns=["model", "psnr", "ssim"]).to_csv(args.out_csv, index=False)
        print(f"\n-> {args.out_csv}")


if __name__ == "__main__":
    main()
