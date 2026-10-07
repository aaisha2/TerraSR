"""Stage 9 ablation table: attribute TerraSR's gain to a cause, or don't claim it.

Evaluates every variant trained by training/run_ablation.py on the held-out
test split and builds the comparison chain, so the write-up can say which part
of the method produced which part of the gain:

    terrasr_full      - swinir_l1                 the headline claim
    terrasr_full      - terrasr_shuffled_labels   does terrain INFORMATION help?
    terrasr_l1        - swinir_l1                 the conditioning alone
    swinir_terrainloss- swinir_l1                 the loss alone
    swinir_terrainloss- swinir_uniform_loss       per-terrain weighting vs the
                                                  composite loss shape

Each variant is evaluated under the label condition it was TRAINED with (read
from its checkpoint), because a shuffled-label control measured with real
labels is not the model that was trained.

Parameter counts are reported alongside, because a conditioned model has more
of them: a gain that tracks parameter count rather than terrain is not a
terrain result.

With several seeds per variant the spread across seeds is reported, and
differences smaller than that spread are marked as not interpretable. With one
seed, the noise floor from configs/ablation.yaml is used and the table says so.

Usage:
    python evaluation/eval_ablation.py --runs-root checkpoints/ablation \
        --test-csv data/dataset/test.csv --report results/ablation.md
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation._eval_common import (checkpoint_training_conditions,  # noqa: E402
                                      load_model_from_checkpoint,
                                      make_test_loader, run_sr_over_loader)

# (name, left, right, what the difference isolates)
COMPARISONS = [
    ("headline claim", "terrasr_full", "swinir_l1",
     "the full method against the baseline it is sold against"),
    ("terrain information", "terrasr_full", "terrasr_shuffled_labels",
     "identical capacity and loss, real labels vs permuted - THE control"),
    ("terrain information (constant-label form)", "terrasr_full",
     "terrasr_unknown_labels",
     "conditioning present but carrying no per-sample signal"),
    ("conditioning alone", "terrasr_l1", "swinir_l1",
     "FiLM terrain conditioning, plain L1 both sides"),
    ("loss alone", "swinir_terrainloss", "swinir_l1",
     "terrain-aware loss, no architectural conditioning"),
    ("per-terrain weighting", "swinir_terrainloss", "swinir_uniform_loss",
     "same composite loss, per-terrain weights vs identical weights"),
]


def find_runs(root: Path, which="best.pth") -> dict:
    """{variant: [(seed, checkpoint_path)]} for every run under root."""
    runs = defaultdict(list)
    for ckpt in sorted(root.glob(f"*/{which}")):
        name = ckpt.parent.name
        m = re.match(r"^(.*)_seed(\d+)$", name)
        variant, seed = (m.group(1), int(m.group(2))) if m else (name, None)
        runs[variant].append((seed, ckpt))
    return dict(runs)


def evaluate_run(ckpt: Path, args, device) -> dict:
    cond = checkpoint_training_conditions(ckpt)
    model, name, is_terrain = load_model_from_checkpoint(ckpt, device)
    loader = make_test_loader(args.test_csv, args.terrain_config,
                               batch_size=args.batch_size,
                               pan_filter=args.pan_filter)
    # evaluate under the trained label condition; a fixed seed keeps the
    # shuffled control reproducible between report runs
    torch.manual_seed(0)
    with torch.no_grad():
        df = run_sr_over_loader(loader, device, model=model, is_terrain=is_terrain,
                                 with_lpips=args.with_lpips,
                                 label_mode=cond["label_mode"])
    out = dict(cond)
    out.update({"checkpoint": str(ckpt), "n_test": len(df),
                "psnr": float(df["psnr"].mean()), "ssim": float(df["ssim"].mean())})
    if "lpips" in df.columns:
        out["lpips"] = float(df["lpips"].mean())
    out["_per_terrain"] = (df.groupby("terrain")["psnr"].mean().to_dict()
                           if "terrain" in df.columns else {})
    return out


def aggregate(runs_by_variant: dict) -> pd.DataFrame:
    rows = []
    for variant, runs in runs_by_variant.items():
        psnr = np.array([r["psnr"] for r in runs])
        ssim = np.array([r["ssim"] for r in runs])
        row = {
            "variant": variant, "n_seeds": len(runs),
            "psnr": float(psnr.mean()),
            "psnr_spread": float(psnr.max() - psnr.min()) if len(psnr) > 1 else np.nan,
            "ssim": float(ssim.mean()),
            "n_params": runs[0]["n_params"],
            "conditioned": runs[0]["conditioned"],
            "loss_mode": runs[0]["loss_mode"],
            "label_mode": runs[0]["label_mode"],
            "epoch_reached": runs[0]["epoch_reached"],
            "epochs_planned": runs[0]["epochs"],
        }
        if all("lpips" in r for r in runs):
            row["lpips"] = float(np.mean([r["lpips"] for r in runs]))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("psnr", ascending=False).reset_index(drop=True)


def noise_floor(table: pd.DataFrame, configured: float) -> tuple:
    """The smallest PSNR difference worth interpreting, and where it came from.
    Measured seed spread beats an assumption whenever it is available."""
    spreads = table["psnr_spread"].dropna()
    if len(spreads):
        return float(spreads.max()), (
            f"largest spread across seeds ({len(spreads)} multi-seed variant(s))")
    return configured, "configured noise_floor_db - NO seed repeats were run"


def build_comparisons(table: pd.DataFrame, floor: float) -> list:
    by_variant = table.set_index("variant")
    out = []
    for label, left, right, meaning in COMPARISONS:
        if left not in by_variant.index or right not in by_variant.index:
            out.append({"comparison": label, "status": "not run",
                         "detail": f"needs both {left} and {right}",
                         "meaning": meaning})
            continue
        l, r = by_variant.loc[left], by_variant.loc[right]
        d_psnr = l["psnr"] - r["psnr"]
        d_ssim = l["ssim"] - r["ssim"]
        out.append({
            "comparison": label, "left": left, "right": right, "meaning": meaning,
            "delta_psnr_db": float(d_psnr), "delta_ssim": float(d_ssim),
            "delta_params": int(l["n_params"] - r["n_params"]),
            "interpretable": bool(abs(d_psnr) > floor),
            "status": ("supported" if d_psnr > floor else
                        "contradicted" if d_psnr < -floor else
                        "within noise"),
        })
    return out


def render(table: pd.DataFrame, comps: list, floor: float, floor_src: str,
            per_terrain: dict) -> str:
    L = ["# Stage 8 ablation - what is the gain actually from?", "",
         "TerraSR changes the architecture (FiLM terrain conditioning) and the",
         "loss (terrain-weighted composite) at the same time. Comparing it only",
         "against SwinIR + L1 cannot attribute a gain to terrain: the same",
         "number would appear if it came from the extra parameters, or from the",
         "composite loss shape with terrain contributing nothing. This grid",
         "separates those.", "",
         f"Smallest interpretable PSNR difference: **{floor:.3f} dB** ({floor_src}).", ""]

    if "NO seed repeats" in floor_src:
        L += ["> Only one seed per variant was trained, so this floor is an",
              "> assumption rather than a measurement. Any difference close to it",
              "> needs a second seed before it goes in the write-up.", ""]

    L += ["## Variants", "",
          "| variant | test PSNR (dB) | spread | SSIM | params | conditioning | loss | labels | epochs |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        spread = "-" if pd.isna(r["psnr_spread"]) else f"{r['psnr_spread']:.3f}"
        L.append(f"| `{r['variant']}` | {r['psnr']:.3f} | {spread} | {r['ssim']:.4f} | "
                 f"{r['n_params']/1e6:.3f}M | {'on' if r['conditioned'] else 'off'} | "
                 f"{r['loss_mode']} | {r['label_mode']} | "
                 f"{r['epoch_reached']}/{r['epochs_planned']} |")
    L.append("")

    L += ["## Attribution", "",
          "| what it isolates | comparison | ΔPSNR (dB) | ΔSSIM | Δparams | verdict |",
          "|---|---|---|---|---|---|"]
    for c in comps:
        if c["status"] == "not run":
            L.append(f"| {c['comparison']} | - | - | - | - | not run ({c['detail']}) |")
            continue
        L.append(f"| {c['comparison']} | `{c['left']}` - `{c['right']}` | "
                 f"{c['delta_psnr_db']:+.3f} | {c['delta_ssim']:+.4f} | "
                 f"{c['delta_params']:+,} | {c['status']} |")
    L.append("")

    for c in comps:
        if c["status"] != "not run":
            L.append(f"- **{c['comparison']}** ({c['meaning']}): "
                     f"{c['delta_psnr_db']:+.3f} dB - {c['status']}.")
    L.append("")

    control = next((c for c in comps if c["comparison"] == "terrain information"), None)
    L += ["## How to read the control", ""]
    if control and control["status"] == "not run":
        L += ["`terrasr_shuffled_labels` was not trained, so the central question -",
              "whether the gain comes from terrain information rather than from the",
              "extra parameters and the composite loss - is still unanswered. Train",
              "it before claiming a terrain effect:", "",
              "```bash",
              "python training/run_ablation.py --only terrasr_shuffled_labels",
              "```", ""]
    elif control:
        if control["status"] == "supported":
            L += [f"`terrasr_full` beats the shuffled-label control by "
                  f"{control['delta_psnr_db']:+.3f} dB with identical capacity and",
                  "loss. That difference is attributable to terrain information,",
                  "because nothing else differs between the two runs.", ""]
        elif control["status"] == "within noise":
            L += [f"`terrasr_full` is within noise of the shuffled-label control "
                  f"({control['delta_psnr_db']:+.3f} dB).",
                  "The two runs differ only in whether each patch gets its own",
                  "terrain label, so on this evidence the conditioning is not using",
                  "terrain information - whatever gain the full model shows over the",
                  "baseline comes from the added capacity and the composite loss.",
                  "This is a result worth reporting plainly, not a reason to drop",
                  "the control.", ""]
        else:
            L += [f"`terrasr_full` is WORSE than the shuffled-label control "
                  f"({control['delta_psnr_db']:+.3f} dB), which means the real",
                  "labels actively hurt. Check label quality first (stage 4,",
                  "label_quality_report.py): conditioning on noisy labels can cost",
                  "more than it gains.", ""]

    if per_terrain:
        terrains = sorted({t for d in per_terrain.values() for t in d})
        if terrains:
            L += ["## Per-terrain test PSNR (dB)", "",
                  "Terrain conditioning should help most where terrain differs most.",
                  "An empty column is a terrain with no test patches - see the stage 4",
                  "label quality report.", "",
                  "| variant | " + " | ".join(terrains) + " |",
                  "|---" * (len(terrains) + 1) + "|"]
            for variant, d in per_terrain.items():
                cells = [(f"{d[t]:.2f}" if t in d else "-") for t in terrains]
                L.append(f"| `{variant}` | " + " | ".join(cells) + " |")
            L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", type=Path, default=Path("checkpoints/ablation"))
    ap.add_argument("--test-csv", required=True, type=Path)
    ap.add_argument("--terrain-config", default=Path("configs/terrain_classes.yaml"), type=Path)
    ap.add_argument("--ablation-config", default=Path("configs/ablation.yaml"), type=Path)
    ap.add_argument("--which", default="best.pth", choices=["best.pth", "last.pth"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--pan-filter", default="all")
    ap.add_argument("--with-lpips", action="store_true")
    ap.add_argument("--report", type=Path, default=None)
    args = ap.parse_args()

    runs = find_runs(args.runs_root, args.which)
    if not runs:
        raise SystemExit(f"no {args.which} checkpoints under {args.runs_root} - "
                          f"run training/run_ablation.py first")

    floor_cfg = 0.10
    if args.ablation_config.exists():
        floor_cfg = float(yaml.safe_load(args.ablation_config.read_text())
                          .get("noise_floor_db", 0.10))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} | variants: {len(runs)} | test: {args.test_csv}")

    evaluated, per_terrain = {}, {}
    for variant, items in sorted(runs.items()):
        evaluated[variant] = []
        for seed, ckpt in items:
            print(f"  {variant}" + (f" (seed {seed})" if seed is not None else "")
                  + " ...", end="", flush=True)
            r = evaluate_run(ckpt, args, device)
            evaluated[variant].append(r)
            print(f" PSNR {r['psnr']:.3f} dB, SSIM {r['ssim']:.4f} "
                  f"[{r['loss_mode']}, labels={r['label_mode']}, "
                  f"{r['n_params']/1e6:.3f}M params]")
        per_terrain[variant] = evaluated[variant][0]["_per_terrain"]

    table = aggregate(evaluated)
    floor, floor_src = noise_floor(table, floor_cfg)
    comps = build_comparisons(table, floor)

    print(f"\nsmallest interpretable PSNR difference: {floor:.3f} dB ({floor_src})\n")
    print(f"{'what it isolates':<44}{'dPSNR':>9}  verdict")
    print("-" * 78)
    for c in comps:
        if c["status"] == "not run":
            print(f"{c['comparison']:<44}{'-':>9}  not run ({c['detail']})")
        else:
            print(f"{c['comparison']:<44}{c['delta_psnr_db']:>+9.3f}  {c['status']}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render(table, comps, floor, floor_src, per_terrain),
                                encoding="utf-8")
        payload = {"noise_floor_db": floor, "noise_floor_source": floor_src,
                   "variants": json.loads(table.to_json(orient="records")),
                   "comparisons": comps,
                   "per_terrain_psnr": per_terrain}
        args.report.with_suffix(".json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8")
        table.to_csv(args.report.with_suffix(".csv"), index=False)
        print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()
