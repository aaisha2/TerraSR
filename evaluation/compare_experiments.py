"""Compare Experiment 1 (true PAN only) vs Experiment 2 (true PAN + pseudo-PAN).

Answers the actual research question: does adding grayscale-converted RGB
imagery as extra training data help, hurt, or do nothing for a SwinIR-based
satellite SR model?

Reports PSNR, SSIM, LPIPS and inference speed for each arm, plus the delta,
and writes a shareable .docx/.md report.

METHODOLOGICAL NOTE (important for the write-up): both arms are evaluated on
the SAME test set, and that test set is filtered to TRUE PAN only by default
(--eval-pan-filter true_pan_only). This is deliberate — the deployment target
is real panchromatic imagery, so measuring both arms on true PAN is the only
comparison that answers "does pseudo-PAN augmentation improve real-PAN
performance?". Evaluating arm 2 on a mixed test set would let it score well by
being good at pseudo-PAN, which is not the goal.

Usage:
    python evaluation/compare_experiments.py \
        --exp1-checkpoint checkpoints/exp1_pan_only/terrasr/best.pth \
        --exp2-checkpoint checkpoints/exp2_pan_plus_pseudo/terrasr/best.pth \
        --test-csv data/dataset/test.csv \
        --out data/dataset/experiment_comparison
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation._eval_common import (load_model_from_checkpoint,  # noqa: E402
                                       make_test_loader, measure_inference_speed,
                                       run_sr_over_loader)
from training.train_utils import get_device  # noqa: E402


def evaluate(label, ckpt, loader, device, lr_size):
    model, name, is_terrain = load_model_from_checkpoint(ckpt, device)
    df = run_sr_over_loader(loader, device, model=model, is_terrain=is_terrain,
                             with_lpips=True)
    speed = measure_inference_speed(model, is_terrain, device, lr_size=lr_size)
    row = {
        "experiment": label,
        "model": name,
        "checkpoint": str(ckpt),
        "psnr": df["psnr"].mean(),
        "ssim": df["ssim"].mean(),
        "lpips": df["lpips"].mean() if "lpips" in df.columns else float("nan"),
        "ms_per_image": speed["ms_per_image"],
        "images_per_sec": speed["images_per_sec"],
        "n_test": len(df),
    }
    return row, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp1-checkpoint", required=True, type=Path,
                     help="model trained on true PAN only")
    ap.add_argument("--exp2-checkpoint", required=True, type=Path,
                     help="model trained on true PAN + pseudo-PAN")
    ap.add_argument("--test-csv", required=True, type=Path)
    ap.add_argument("--terrain-config", default="configs/terrain_classes.yaml")
    ap.add_argument("--eval-pan-filter", default="true_pan_only",
                     choices=["true_pan_only", "all", "pseudo_pan_only"],
                     help="what to evaluate BOTH arms on (default: true_pan_only — "
                          "see the methodological note in this file's docstring)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-size", type=int, default=128, help="LR size for the speed benchmark")
    ap.add_argument("--out", type=Path, default=None,
                     help="path WITHOUT extension for the report (.md/.docx)")
    args = ap.parse_args()

    device = get_device()
    loader = make_test_loader(args.test_csv, args.terrain_config, args.batch_size,
                              pan_filter=args.eval_pan_filter)
    print(f"device: {device} | test patches: {len(loader.dataset)} "
          f"(pan_filter={args.eval_pan_filter})\n")

    rows, per_patch = [], {}
    for label, ckpt in [("Exp1: true PAN only", args.exp1_checkpoint),
                         ("Exp2: true PAN + pseudo-PAN", args.exp2_checkpoint)]:
        print(f"evaluating {label} ...")
        row, df = evaluate(label, ckpt, loader, device, args.lr_size)
        rows.append(row)
        per_patch[label] = df

    summary = pd.DataFrame(rows)
    print(f"\n{'experiment':<30}{'PSNR':>9}{'SSIM':>9}{'LPIPS':>9}{'ms/img':>9}{'img/s':>9}")
    print("-" * 75)
    for r in rows:
        print(f"{r['experiment']:<30}{r['psnr']:>9.3f}{r['ssim']:>9.4f}"
              f"{r['lpips']:>9.4f}{r['ms_per_image']:>9.1f}{r['images_per_sec']:>9.1f}")

    a, b = rows[0], rows[1]
    d_psnr = b["psnr"] - a["psnr"]
    d_ssim = b["ssim"] - a["ssim"]
    d_lpips = b["lpips"] - a["lpips"]
    print(f"\n{'delta (Exp2 - Exp1)':<30}{d_psnr:>+9.3f}{d_ssim:>+9.4f}{d_lpips:>+9.4f}")
    print("  (PSNR/SSIM: higher is better.  LPIPS: LOWER is better, so a "
          "negative delta means Exp2 is perceptually better.)")

    verdict = []
    verdict.append(f"PSNR {'improved' if d_psnr > 0 else 'degraded'} by {abs(d_psnr):.3f} dB")
    verdict.append(f"SSIM {'improved' if d_ssim > 0 else 'degraded'} by {abs(d_ssim):.4f}")
    verdict.append(f"LPIPS {'improved' if d_lpips < 0 else 'degraded'} by {abs(d_lpips):.4f}")
    print("\nverdict: adding pseudo-PAN — " + "; ".join(verdict) + ".")

    # per-terrain delta, so a mixed overall result can be attributed
    print("\nper-terrain PSNR delta (Exp2 - Exp1):")
    pt_a = per_patch[a["experiment"]].groupby("terrain")["psnr"].mean()
    pt_b = per_patch[b["experiment"]].groupby("terrain")["psnr"].mean()
    for terrain in sorted(set(pt_a.index) | set(pt_b.index)):
        if terrain in pt_a.index and terrain in pt_b.index:
            print(f"  {terrain:<22}{pt_b[terrain] - pt_a[terrain]:+.3f} dB")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out.with_suffix(".csv"), index=False)
        lines = ["# Experiment comparison: true PAN vs true PAN + pseudo-PAN", "",
                 f"Test patches: {a['n_test']} (pan_filter=`{args.eval_pan_filter}`) "
                 f"· device: {device}", "",
                 "| Experiment | PSNR (dB) | SSIM | LPIPS | ms/img | img/s |",
                 "| --- | --- | --- | --- | --- | --- |"]
        for r in rows:
            lines.append(f"| {r['experiment']} | {r['psnr']:.3f} | {r['ssim']:.4f} | "
                          f"{r['lpips']:.4f} | {r['ms_per_image']:.1f} | {r['images_per_sec']:.1f} |")
        lines += ["", f"**Delta (Exp2 − Exp1):** PSNR {d_psnr:+.3f} dB · SSIM {d_ssim:+.4f} · "
                       f"LPIPS {d_lpips:+.4f} (LPIPS: lower is better)", "",
                   "_Both arms are evaluated on the same true-PAN test set, because the "
                   "deployment target is real panchromatic imagery._"]
        args.out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
        print(f"\n-> {args.out.with_suffix('.md')}")
        print(f"-> {args.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
