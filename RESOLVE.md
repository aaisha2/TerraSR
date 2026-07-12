# Running TerraSR on the RESOLVE workstation

End-to-end runbook: environment → complete download → full data build →
training → evaluation. Everything is driven from the repo root.

The design principle (docs/plan §8): develop/smoke-test on the laptop against
tiny samples, then do the heavy download + train **on RESOLVE**. Downloaded
imagery and all intermediates live under `data/` (gitignored) — only code and
configs are in git, so RESOLVE just pulls the repo and builds `data/` locally.

---

## 1. Environment

RESOLVE has a GPU, so install a **CUDA** build of PyTorch first (matched to the
box's CUDA version — check with `nvidia-smi`), then the rest:

```bash
git clone <repo-url> terrasr && cd terrasr
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate

# CUDA torch FIRST (example for CUDA 12.1 — adjust cuNNN to the box):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`rasterio` ships its own GDAL wheel — no system GDAL needed. If a corporate
proxy blocks the torch hub, run the weight prefetch (step 4) from a node that
can reach `download.pytorch.org`.

**Disk:** the complete SpaceNet PAN set alone is ~130 GB (6 AOIs, ~190 COG
scenes); Maxar events add more. Budget **~300–400 GB** for `data/` (raw +
standardized + patches + pairs). Point `data/` at a large volume if `/home` is
small (symlink `data/` or edit `dirs:` in `configs/pipeline.yaml`).

---

## 2. Preview the download (no fetch)

Confirm what the complete pull will be before committing the disk/bandwidth:

```bash
python run_pipeline.py --stages download --list-only
```

This lists every SpaceNet PAN scene and Maxar `pan_analytic` tile in the
configured AOIs/events, with sizes and a grand total. Edit
`configs/datasets.yaml` to add/remove AOIs (`spacenet.aois`) or events
(`maxar.events`) — the primary, required sources are SpaceNet + Maxar (true
PAN). OpenEarthMap/NAIP are supplementary pseudo-PAN and are left un-wired
(commented in `configs/pipeline.yaml`) until their downloaders are added.

---

## 3. Full data build (stages 1–6), one command

```bash
python run_pipeline.py
```

Runs, in order, on the real `data/` tree:

| stage | script | output |
|-------|--------|--------|
| download    | `download_spacenet.py`, `download_maxar.py` | `data/raw/<source>/` |
| standardize | `standardize_scenes.py` (per source, correct PAN mode) | `data/standardized/<source>/` |
| patchify    | `tile_extractor.py --recursive` | `data/patches/` |
| filter      | `patch_filter.py` | `patch_manifest_filtered.json` |
| label       | `worldcover_zonal_stats.py` + `assign_dominant_terrain.py` | `patch_manifest_labeled.json` |
| degrade     | `make_lr_hr_pairs.py` (16-bit GeoTIFF) | `data/pairs/` |
| package     | `build_manifest.py` + `split_train_val_test.py` + `dataset_stats.py` | `data/dataset/` |

Every stage is **idempotent** — downloads skip files already on disk (matched
by size), standardize skips scenes already converted — so a re-run resumes
rather than redoing work. Run subsets or resume with `--stages` / `--skip`:

```bash
python run_pipeline.py --stages download          # just the complete download
python run_pipeline.py --skip download            # everything after download
python run_pipeline.py --stages label,degrade,package
```

At the end, read the `dataset_stats.py` report: it prints per-terrain counts
and flags any terrain below the per-class floor (the water/mountain scarcity
risk). If water/mountain are short, add targeted Maxar events to
`configs/datasets.yaml` and re-run `--stages download,standardize` then
`--skip download`.

**Before trusting the LR for training** run the supervisor-required validation
(stage 5) against real satellite LR reference chips and tune
`configs/degradation.yaml` if needed:

```bash
python data_pipeline/05_degrade/validate_against_real_lr.py \
    --synthetic-dir data/pairs/lr --real-dir <real-LR-chips>
```

---

## 4. Prefetch pretrained weights (optional but recommended)

Caches VGG16 (perceptual loss) and Faster R-CNN (detection harness) up front so
training/eval don't stall mid-run:

```bash
python scripts/prefetch_weights.py
```

---

## 5. Training + evaluation (stages 7–9)

Point the train configs at the built dataset (they default to
`data/dataset/{train,val}.csv`, which is exactly what stage 6 produced) and
train. Either chain them via the orchestrator:

```bash
python run_pipeline.py --skip download --with-training   # data build already done -> just 7-9
```

or run each explicitly (recommended first time, so you watch each converge):

```bash
# baselines
python training/train_baseline.py --config configs/train_baseline.yaml --override model.name=srcnn  train.out_dir=checkpoints/srcnn
python training/train_baseline.py --config configs/train_baseline.yaml --override model.name=srgan  train.out_dir=checkpoints/srgan
python training/train_baseline.py --config configs/train_baseline.yaml --override model.name=swinir train.out_dir=checkpoints/swinir

# TerraSR (terrain embedding + terrain-aware loss)
python training/train_terrasr.py --config configs/train_terrasr.yaml

# evaluation — all four models + the bicubic floor (transformer vs CNN vs GAN
# vs terrain-aware). TerraSR is passed LAST so the per-terrain delta reads
# terrasr - each baseline.
python evaluation/eval_psnr_ssim.py  --test-csv data/dataset/test.csv --with-bicubic \
    --checkpoints checkpoints/srcnn/best.pth checkpoints/srgan/best.pth \
                  checkpoints/swinir/best.pth checkpoints/terrasr/best.pth
python evaluation/eval_per_terrain.py --test-csv data/dataset/test.csv \
    --checkpoints checkpoints/srcnn/best.pth checkpoints/srgan/best.pth \
                  checkpoints/swinir/best.pth checkpoints/terrasr/best.pth
```

Training auto-uses the GPU when `torch.cuda.is_available()`. Tune
`train.batch_size`, `train.epochs`, and `model.*` (full-size SwinIR:
`embed_dim: 60`, `depths: [6,6,6,6]`) in the train configs or via `--override`.

---

## Quick reference

```bash
python run_pipeline.py --list-only                # preview download size
python run_pipeline.py                            # full data build (1-6)
python run_pipeline.py --with-training            # + train + eval (1-9)
python run_pipeline.py --stages <subset>          # run/resume specific stages
```

Paths, per-source PAN modes, and stage configs all live in
`configs/pipeline.yaml`.
