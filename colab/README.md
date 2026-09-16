# TerraSR on Google Colab

Run the full TerraSR super-resolution pipeline on a free Google Colab GPU, with Google Drive as persistent storage for the dataset, checkpoints, and results.

## Quick Start

1. Open the notebook in Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aaisha2/TerraSR/blob/main/colab/TerraSR_Colab.ipynb)

2. In the Colab menu: **Runtime → Change runtime type → T4 GPU** (or better)

3. Run cells top to bottom. Each section is self-contained and idempotent — you can re-run any section safely.

---

## Google Drive Layout

The notebook creates this structure in your Drive on first run:

```
MyDrive/TerraSR-Colab/
├── data/
│   ├── raw/          ← downloaded satellite imagery
│   ├── standardized/ ← stage 2 output
│   ← patches/       ← stage 3/4 patches + manifests
│   ├── pairs/        ← stage 5 LR/HR GeoTIFF pairs
│   ├── dataset/      ← stage 6 manifests + train/val/test CSVs
│   └── cache/        ← ESA WorldCover tile cache (~110 MB/tile)
├── checkpoints/      ← last.pth + best.pth per model (safe across runtime resets)
├── results/          ← evaluation outputs, PSNR/SSIM tables, .docx report
└── logs/             ← training stdout (redirected in Section 11)
```

The notebook symlinks `/content/TerraSR/data` → Drive and `/content/TerraSR/checkpoints` → Drive, so all relative paths in the research code resolve without any config changes.

---

## Four-Phase Workflow

### Phase 1 — Infrastructure test (Sections 0–8, ~30 min)
- GPU check, Drive mount, repo clone, environment install
- Download a tiny subset (3 Vegas scenes + 2 Maxar tiles)
- Run stages 2–6 to build a small dataset
- Run one batch forward/backward to confirm GPU works end-to-end

### Phase 2 — Smoke training (Section 9, ~15 min)
- 5 epochs, batch_size=4, perceptual loss disabled
- Confirms: loss decreases, checkpoints appear on Drive, resume works

### Phase 3 — Scaled experiment (Sections 10–11)
- Increase AOIs/events in Section 5 and re-run stages 2–6
- Run full 100-epoch training for all four models overnight
- Checkpoints are safe on Drive even if the Colab runtime terminates

### Phase 4 — Evaluation (Section 12)
- PSNR/SSIM table, per-terrain breakdown, `.docx` report
- Download results to local machine (Section 13)

---

## Resuming After a Runtime Reset

Colab free runtimes disconnect after ~12 hours or on inactivity. To resume:

1. Reconnect and open the notebook
2. Re-run **Sections 0–4** (GPU check → imports; fast, < 2 min)
3. Skip to **Section 11** — training auto-resumes from `last.pth` on Drive

The checkpoint resume logic is built into `train_terrasr.py` and `train_baseline.py` — they detect `last.pth` and continue from the next epoch automatically.

---

## Scaling Up the Dataset

To add more data, edit the download cell in **Section 5**:

```python
# SpaceNet — add more AOIs:
SPACENET_AOIS  = ["AOI_2_Vegas", "AOI_5_Khartoum", "AOI_3_Paris"]
SPACENET_MAX_FILES = None   # None = download everything

# Maxar — add more events:
MAXAR_EVENTS   = ["Brazil-Flooding-May24", "Belize-Wildfires-June24"]
MAXAR_MAX_FILES = None
```

Then re-run Sections 5–6 (the pipeline scripts are idempotent — already-downloaded files are skipped).

> **Storage note:** The full dataset is ~300–400 GB. With Google AI Pro (5 TB Drive), storage is not the bottleneck. Colab's local VM disk (~200 GB) fills up faster — keep the Drive symlinks in place so data lands on Drive, not the VM.

---

## Batch Size Guide

| Colab GPU | Recommended batch_size |
|---|---|
| T4 (15 GB) | 8 |
| L4 (22 GB) | 12–16 |
| A100 (40 GB) | 16 |

The notebook defaults to `batch_size=8`. To use 16, edit the override in Section 11.

---

## Relationship to the Research Codebase

This notebook is a **thin orchestration layer** over the unchanged research code. It does not modify:

- Any model (`terrasr_swinir.py`, `swinir_baseline.py`, etc.)
- Any training script (`train_terrasr.py`, `train_baseline.py`)
- Any config (`train_terrasr.yaml`, `terrain_aware_loss.yaml`, etc.)
- The data pipeline (stages 1–6)
- The evaluation scripts

If you regain access to the RESOLVE workstation, the research code runs there identically — nothing in `colab/` is required for that.
