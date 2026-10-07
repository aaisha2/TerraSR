"""Stage 8 ablation runner: trains every variant in configs/ablation.yaml.

The point of the grid is attribution. TerraSR changes the architecture AND
the loss at once, so `terrasr` vs `swinir+L1` cannot tell terrain apart from
extra capacity or from a better loss shape. This trains the variants that can,
including the shuffled-label control that decides whether terrain information
is doing any work at all.

Each variant is an independent training run with its own checkpoint directory,
so it resumes exactly like any other run: re-running this script continues
unfinished variants and skips finished ones. That matters because the grid is
7 variants x N seeds - far longer than one Colab session.

Usage:
    python training/run_ablation.py                      # every variant, config seeds
    python training/run_ablation.py --only terrasr_full terrasr_shuffled_labels
    python training/run_ablation.py --seeds 42 43        # override the seed list
    python training/run_ablation.py --dry-run            # print the commands
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train_utils import checkpoint_progress  # noqa: E402

PY = sys.executable


def run_dir(root: Path, variant: str, seed: int, n_seeds: int) -> Path:
    """One directory per (variant, seed). A single-seed grid keeps the plain
    variant name so paths stay readable."""
    return root / (variant if n_seeds == 1 else f"{variant}_seed{seed}")


def flat_overrides(d: dict) -> list:
    return [f"{k}={json.dumps(v) if isinstance(v, bool) else v}" for k, v in d.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=Path("configs/ablation.yaml"), type=Path)
    ap.add_argument("--out-root", type=Path, default=Path("checkpoints/ablation"))
    ap.add_argument("--only", nargs="*", default=None,
                     help="run just these variant names")
    ap.add_argument("--seeds", nargs="*", type=int, default=None,
                     help="override the seed list from the config")
    ap.add_argument("--extra-override", nargs="*", default=[],
                     help="dotted overrides applied to every variant, e.g. "
                          "train.epochs=5 train.batch_size=4")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fresh", action="store_true",
                     help="retrain variants from scratch instead of resuming")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    base_config = cfg["base_config"]
    common = cfg.get("common_overrides") or {}
    variants = cfg["variants"]
    seeds = args.seeds if args.seeds else cfg.get("seeds", [42])

    selected = args.only or list(variants)
    unknown = [v for v in selected if v not in variants]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; "
                          f"available: {sorted(variants)}")

    print(f"ablation: {len(selected)} variant(s) x {len(seeds)} seed(s) "
          f"= {len(selected)*len(seeds)} run(s)")
    print(f"base config: {base_config}")
    print(f"out root:    {args.out_root}")
    if len(seeds) == 1:
        print("NOTE: a single seed cannot separate a small real effect from "
              "initialisation noise. Add seeds (configs/ablation.yaml -> seeds, "
              "or --seeds 42 43) before drawing a conclusion from a sub-0.1 dB "
              "difference.")
    print()

    # the terrain-conditioned trainer handles every variant: it can switch the
    # conditioning off, swap the loss, and alter the label mode
    script = Path(__file__).parent / "train_terrasr.py"
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    results, failures = [], []

    for seed in seeds:
        for name in selected:
            spec = variants[name]
            out_dir = run_dir(args.out_root, name, seed, len(seeds))
            overrides = dict(common)
            overrides.update(spec.get("overrides") or {})
            overrides["train.seed"] = seed
            overrides["train.out_dir"] = str(out_dir)

            cmd = [PY, "-u", str(script), "--config", base_config,
                   "--override", *flat_overrides(overrides), *args.extra_override]
            if args.fresh:
                cmd.append("--fresh")

            progress = checkpoint_progress(out_dir)
            done = progress.get("last_epoch")
            total = progress.get("total_epochs")
            state = (f"resuming from epoch {done}" if done and total and done < total
                     else "already finished" if done and total and done >= total
                     else "starting")

            print("=" * 72)
            print(f"{name}  (seed {seed})  [{state}]")
            print(f"  {spec.get('description', '').strip()}")
            print("=" * 72, flush=True)
            if args.dry_run:
                print("  " + " ".join(cmd) + "\n")
                continue

            t0 = time.time()
            rc = subprocess.run(cmd, env=env).returncode
            mins = (time.time() - t0) / 60
            after = checkpoint_progress(out_dir)
            results.append({"variant": name, "seed": seed, "out_dir": str(out_dir),
                             "returncode": rc, "minutes": round(mins, 1),
                             "epoch": after.get("last_epoch"),
                             "best_psnr": after.get("best_metric")})
            if rc != 0:
                failures.append(f"{name} (seed {seed}) exited {rc}")
                print(f"\n{name} seed {seed} FAILED (exit {rc}) - continuing with "
                      f"the rest of the grid; re-run to retry it.\n", flush=True)
            else:
                print(f"\n{name} seed {seed} done in {mins:.1f} min "
                      f"(best val PSNR {after.get('best_metric')})\n", flush=True)

    if args.dry_run:
        return

    summary = args.out_root / "ablation_runs.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("=" * 72)
    print(f"{'variant':<28}{'seed':>5}{'epoch':>7}{'best val PSNR':>15}{'min':>7}")
    for r in results:
        best = r["best_psnr"]
        print(f"{r['variant']:<28}{r['seed']:>5}{str(r['epoch'] or '-'):>7}"
              f"{(f'{best:.2f}' if isinstance(best, float) else '-'):>15}"
              f"{r['minutes']:>7.1f}")
    print(f"\nrun summary -> {summary}")
    print("Now build the attribution table:")
    print(f"  python evaluation/eval_ablation.py --runs-root {args.out_root} "
          f"--test-csv data/dataset/test.csv")

    if failures:
        print("\nfailed runs:")
        for f in failures:
            print(f"  {f}")
        raise SystemExit(f"{len(failures)} ablation run(s) failed")


if __name__ == "__main__":
    main()
