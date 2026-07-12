"""End-to-end pipeline orchestrator (built for the RESOLVE workstation).

Runs the data-build stages 1->6 on the full dataset in one command, using the
real data/ tree defined in configs/pipeline.yaml. Each stage is invoked as a
subprocess of the existing stage script, so the orchestrator is a thin, honest
wrapper around exactly what you'd type by hand — and every stage stays
independently runnable and idempotent.

    python run_pipeline.py                      # full data build, stages 1-6
    python run_pipeline.py --stages download    # just the (complete) download
    python run_pipeline.py --skip download       # everything except download
    python run_pipeline.py --list-only           # dry-run downloads (no fetch)
    python run_pipeline.py --with-training       # also run stages 7-9

Stages 7-9 (baselines, TerraSR, evaluation) are heavy and GPU-bound; they run
only with --with-training. See RESOLVE.md for the full runbook.
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

PY = sys.executable
STAGE_ORDER = ["download", "standardize", "patchify", "filter", "label",
               "degrade", "package"]


def run(cmd, dry=False):
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}")
    if dry:
        return
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        raise SystemExit(f"stage failed (exit {result.returncode}): {printable}")


def stage_download(cfg, dirs, list_only):
    for name, src in cfg["sources"].items():
        script = src.get("download_script")
        if not script or not Path(script).exists():
            print(f"  [skip] {name}: no download script wired yet")
            continue
        cmd = [PY, script, "--config", cfg["configs"]["datasets"]]
        if list_only:
            cmd.append("--list-only")
        run(cmd)


def stage_standardize(cfg, dirs):
    for name, src in cfg["sources"].items():
        raw = Path(dirs["raw"]) / name
        if not raw.exists():
            print(f"  [skip] {name}: {raw} not present")
            continue
        cmd = [PY, "data_pipeline/02_standardize/standardize_scenes.py",
               "--in-dir", raw, "--out-dir", Path(dirs["standardized"]) / name,
               "--mode", src["mode"]]
        if src.get("recursive"):
            cmd.append("--recursive")
        run(cmd)


def stage_patchify(cfg, dirs):
    run([PY, "data_pipeline/03_patchify/tile_extractor.py",
         "--in-dir", dirs["standardized"], "--recursive",
         "--out-dir", dirs["patches"], "--config", cfg["configs"]["patchify"]])


def stage_filter(cfg, dirs):
    run([PY, "data_pipeline/03_patchify/patch_filter.py",
         "--manifest", Path(dirs["patches"]) / "patch_manifest.json",
         "--config", cfg["configs"]["patchify"]])


def stage_label(cfg, dirs):
    patches = Path(dirs["patches"])
    run([PY, "data_pipeline/04_labeling/worldcover_zonal_stats.py",
         "--manifest", patches / "patch_manifest_filtered.json",
         "--config", cfg["configs"]["terrain"], "--only-kept"])
    run([PY, "data_pipeline/04_labeling/assign_dominant_terrain.py",
         "--manifest", patches / "patch_manifest_zonal.json",
         "--config", cfg["configs"]["terrain"]])


def stage_degrade(cfg, dirs):
    run([PY, "data_pipeline/05_degrade/make_lr_hr_pairs.py",
         "--manifest", Path(dirs["patches"]) / "patch_manifest_labeled.json",
         "--out-dir", dirs["pairs"], "--config", cfg["configs"]["degradation"],
         "--only-labeled"])


def stage_package(cfg, dirs):
    dataset = Path(dirs["dataset"])
    run([PY, "data_pipeline/06_package/build_manifest.py",
         "--labeled", Path(dirs["patches"]) / "patch_manifest_labeled.json",
         "--degradation", Path(dirs["pairs"]) / "degradation_manifest.json",
         "--out-dir", dataset])
    run([PY, "data_pipeline/06_package/split_train_val_test.py",
         "--manifest", dataset / "dataset_manifest.parquet",
         "--config", cfg["configs"]["split"]])
    run([PY, "data_pipeline/06_package/dataset_stats.py",
         "--manifest", dataset / "dataset_manifest_split.parquet",
         "--config", cfg["configs"]["split"]])


def stage_training():
    """Stages 7-9. Trains the three baselines + TerraSR, then evaluates all of
    them (transformer vs CNN vs GAN vs terrain-aware). TerraSR is passed LAST to
    the evaluators so the per-terrain delta reads terrasr - each baseline."""
    for name in ("srcnn", "srgan", "swinir"):
        run([PY, "training/train_baseline.py", "--config", "configs/train_baseline.yaml",
             "--override", f"model.name={name}", f"train.out_dir=checkpoints/{name}"])
    run([PY, "training/train_terrasr.py", "--config", "configs/train_terrasr.yaml"])

    ckpts = ["checkpoints/srcnn/best.pth", "checkpoints/srgan/best.pth",
             "checkpoints/swinir/best.pth", "checkpoints/terrasr/best.pth"]
    run([PY, "evaluation/eval_psnr_ssim.py", "--test-csv", "data/dataset/test.csv",
         "--with-bicubic", "--checkpoints", *ckpts])
    run([PY, "evaluation/eval_per_terrain.py", "--test-csv", "data/dataset/test.csv",
         "--checkpoints", *ckpts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/pipeline.yaml")
    ap.add_argument("--stages", help="comma-separated subset of: " + ",".join(STAGE_ORDER))
    ap.add_argument("--skip", help="comma-separated stages to skip")
    ap.add_argument("--list-only", action="store_true", help="dry-run the download stage")
    ap.add_argument("--with-training", action="store_true", help="also run stages 7-9")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    dirs = cfg["dirs"]

    selected = args.stages.split(",") if args.stages else list(STAGE_ORDER)
    if args.skip:
        skip = set(args.skip.split(","))
        selected = [s for s in selected if s not in skip]

    dispatch = {
        "download": lambda: stage_download(cfg, dirs, args.list_only),
        "standardize": lambda: stage_standardize(cfg, dirs),
        "patchify": lambda: stage_patchify(cfg, dirs),
        "filter": lambda: stage_filter(cfg, dirs),
        "label": lambda: stage_label(cfg, dirs),
        "degrade": lambda: stage_degrade(cfg, dirs),
        "package": lambda: stage_package(cfg, dirs),
    }

    print(f"pipeline stages: {selected}")
    for stage in selected:
        print(f"\n{'='*60}\n== stage: {stage}\n{'='*60}")
        dispatch[stage]()

    if args.with_training:
        print(f"\n{'='*60}\n== stages 7-9: training + evaluation\n{'='*60}")
        stage_training()

    print("\npipeline complete.")


if __name__ == "__main__":
    main()
