# UC Merced test pipeline

A **self-contained, isolated** harness to run the complete TerraSR pipeline
start-to-finish on the small [UC Merced Land Use dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)
(2100 images, 21 classes, 256×256) — so you can validate every stage locally
without a RESOLVE-scale download.

## Isolation (why this won't touch your RESOLVE run)

- **UC-Merced-specific stages** (`download_ucmerced.py`, `prepare_ucmerced.py`)
  live only in this folder.
- **Shared stages 5–9** (degrade, package, train, eval) are the **real
  production scripts** in `data_pipeline/`, `training/`, `evaluation/`, called
  read-only via subprocess. This harness never edits them.
- **All inputs** are this folder's `configs/`; **all outputs** go under this
  folder's `data/` and `checkpoints/`. Nothing here writes into the main
  `data/` or `checkpoints/`. Editing the RESOLVE configs doesn't change this
  harness and vice-versa (configs are copied, not shared).

So: run this as often as you like to test the pipeline; the RESOLVE code and
data stay untouched.

## Run

```bash
# full: download -> prepare -> degrade -> package -> train (SwinIR + TerraSR) -> evaluate
python test_pipeline_ucmerced/run_test_pipeline.py

# faster (fewer images per class):
python test_pipeline_ucmerced/run_test_pipeline.py --limit-per-class 20

# just build the dataset (no training/eval):
python test_pipeline_ucmerced/run_test_pipeline.py --no-training
```

First run downloads ~317 MB (torchgeo HuggingFace mirror; the official weegee
link is often down) into `data/raw/` and is idempotent (re-runs skip it).

## What each stage maps to

| Pipeline stage | Here |
|---|---|
| 1 download | `download_ucmerced.py` — UC Merced zip from a HF mirror |
| 2 standardize | `prepare_ucmerced.py` — RGB → pseudo-PAN grayscale uint16 GeoTIFF (BT.709, same weights as the real `extract_pan_band.py`), synthetic UTM CRS + 0.3 m transform, tagged `pseudo_pan=true` |
| 3 patchify | (skipped — UC Merced tiles are already 256×256 = one patch each) |
| 4 label | `prepare_ucmerced.py` — terrain from the UC Merced **class** (folder name) via `configs/ucmerced_terrain.yaml`, not ESA WorldCover (UC Merced tiles aren't georeferenced) |
| 5 degrade | **real** `make_lr_hr_pairs.py` |
| 6 package | **real** `build_manifest.py` + `split_train_val_test.py` + `dataset_stats.py` |
| 7–8 train | **real** `train_baseline.py` (SwinIR) + `train_terrasr.py`, with this folder's small-model configs |
| 9 evaluate | **real** `eval_psnr_ssim.py` + `eval_per_terrain.py` |

Each UC Merced image is its own `source_scene`, so stage 6's geographic-block
split becomes a clean per-image split (no leakage). The class→terrain map
covers 5 of the 7 terrains (Urban, Agriculture, Forest, Water,
Grassland/Wetland); Mountain and Bare Land/Desert have no UC Merced class, the
same kind of coverage gap the real dataset has.

## Web app on the test model

```bash
# Windows: set TERRASR_CKPT=...   then run
TERRASR_CKPT=test_pipeline_ucmerced/checkpoints/terrasr/best.pth python webapp/backend/app.py
```

## Notes

- Configs here use **small models + 5 epochs** so the whole thing finishes on
  CPU in minutes. It's a plumbing test, not a benchmark — PSNR numbers here are
  not meaningful. Scale up (`embed_dim: 60`, `depths: [6,6,6,6]`, more epochs)
  or run on a GPU for real numbers.
- `data/` and `checkpoints/` under this folder are gitignored.
