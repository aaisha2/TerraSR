"""UC Merced end-to-end test harness orchestrator.

Runs the COMPLETE pipeline on the small UC Merced dataset so you can verify
every stage start-to-finish locally — without touching or depending on a
RESOLVE-scale download. It is deliberately isolated from the production run:

  - UC-Merced-specific stages (download, prepare) live in this folder.
  - Shared stages 5-9 (degrade, package, train, eval) call the REAL scripts in
    data_pipeline/ | training/ | evaluation/ via subprocess, READ-ONLY — the
    production code is never modified.
  - All inputs are this folder's configs/; all outputs go under this folder's
    data/ and checkpoints/. Nothing here writes into the main data/ or
    checkpoints/.

    python test_pipeline_ucmerced/run_test_pipeline.py                 # full: data + train + eval
    python test_pipeline_ucmerced/run_test_pipeline.py --limit-per-class 20   # faster
    python test_pipeline_ucmerced/run_test_pipeline.py --no-training   # data build only
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HERE = "test_pipeline_ucmerced"
PY = sys.executable

RAW = f"{HERE}/data/raw"
STD = f"{HERE}/data/standardized"
PATCHES = f"{HERE}/data/patches"
PAIRS = f"{HERE}/data/pairs"
DATASET = f"{HERE}/data/dataset"
CFG = f"{HERE}/configs"
CKPT = f"{HERE}/checkpoints"

LABELED = f"{PATCHES}/patch_manifest_labeled.json"


def run(cmd):
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}", flush=True)
    # PYTHONUNBUFFERED so per-epoch training lines show up live even when the
    # output of this script is piped to a file.
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    r = subprocess.run([str(c) for c in cmd], cwd=str(REPO_ROOT), env=env)
    if r.returncode != 0:
        raise SystemExit(f"step failed (exit {r.returncode}): {printable}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-class", type=int, default=None,
                     help="cap images per class for a faster run")
    ap.add_argument("--no-training", action="store_true",
                     help="build the dataset only (skip stages 7-9)")
    args = ap.parse_args()

    print("=" * 60, "\n== UC Merced test pipeline\n", "=" * 60)

    # --- stage 1: download (UC-Merced-specific) ---
    run([PY, f"{HERE}/download_ucmerced.py", "--out-dir", RAW])

    # --- stages 2-4: prepare (UC-Merced-specific: pseudo-PAN + class label) ---
    prepare = [PY, f"{HERE}/prepare_ucmerced.py", "--raw-dir", RAW,
               "--out-standardized", STD, "--out-manifest", LABELED,
               "--config", f"{CFG}/ucmerced_terrain.yaml"]
    if args.limit_per_class:
        prepare += ["--limit-per-class", args.limit_per_class]
    run(prepare)

    # --- stage 5: degrade (REAL script) ---
    run([PY, "data_pipeline/05_degrade/make_lr_hr_pairs.py",
         "--manifest", LABELED, "--out-dir", PAIRS,
         "--config", f"{CFG}/degradation.yaml", "--only-labeled"])

    # --- stage 6: package (REAL scripts) ---
    run([PY, "data_pipeline/06_package/build_manifest.py",
         "--labeled", LABELED, "--degradation", f"{PAIRS}/degradation_manifest.json",
         "--out-dir", DATASET])
    run([PY, "data_pipeline/06_package/split_train_val_test.py",
         "--manifest", f"{DATASET}/dataset_manifest.parquet", "--config", f"{CFG}/split.yaml"])
    run([PY, "data_pipeline/06_package/dataset_stats.py",
         "--manifest", f"{DATASET}/dataset_manifest_split.parquet", "--config", f"{CFG}/split.yaml"])

    if args.no_training:
        print("\ndata build complete (training skipped).")
        return

    # --- stages 7-8: train the three baselines (SRCNN/SRGAN/SwinIR) + TerraSR
    #     (REAL scripts, test configs). Baselines share train_baseline.yaml;
    #     model.name + out_dir are overridden per model. ---
    for name in ("srcnn", "srgan", "swinir"):
        run([PY, "training/train_baseline.py", "--config", f"{CFG}/train_baseline.yaml",
             "--override", f"model.name={name}", f"train.out_dir={CKPT}/{name}"])
    run([PY, "training/train_terrasr.py", "--config", f"{CFG}/train_terrasr.yaml"])

    # --- stage 9: evaluate all four + bicubic (REAL scripts). TerraSR last so
    #     the per-terrain delta reads terrasr - each baseline. ---
    test_csv = f"{DATASET}/test.csv"
    terrain_cfg = f"{CFG}/ucmerced_terrain.yaml"
    ckpts = [f"{CKPT}/srcnn/best.pth", f"{CKPT}/srgan/best.pth",
             f"{CKPT}/swinir/best.pth", f"{CKPT}/terrasr/best.pth"]
    run([PY, "evaluation/eval_psnr_ssim.py", "--test-csv", test_csv, "--with-bicubic",
         "--terrain-config", terrain_cfg, "--checkpoints", *ckpts])
    run([PY, "evaluation/eval_per_terrain.py", "--test-csv", test_csv,
         "--terrain-config", terrain_cfg, "--checkpoints", *ckpts])

    # --- results report (downloadable .docx + .md) ---
    run([PY, "evaluation/make_results_report.py", "--test-csv", test_csv,
         "--terrain-config", terrain_cfg,
         "--split-manifest", f"{DATASET}/dataset_manifest_split.csv",
         "--title", "TerraSR - UC Merced Pipeline Results",
         "--out", f"{DATASET}/results_report", "--checkpoints", *ckpts])

    print("\nUC Merced test pipeline complete.")
    print(f"  dataset: {DATASET}/  |  checkpoints: {CKPT}/")
    print(f"  results report: {DATASET}/results_report.docx (+ .md)")
    print(f"  web app: TERRASR_CKPT={CKPT}/terrasr/best.pth python webapp/backend/app.py")


if __name__ == "__main__":
    main()
