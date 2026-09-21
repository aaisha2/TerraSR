# TerraSR on Google Colab

Run the full TerraSR super-resolution pipeline on a Google Colab GPU. The pipeline runs on Colab's fast local disk; Google Drive holds only what must survive a runtime reset (checkpoints, results, and one dataset archive).

## Quick Start

1. Open the notebook in Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aaisha2/TerraSR/blob/master/colab/TerraSR_Colab.ipynb)

2. In the Colab menu: **Runtime → Change runtime type → T4 GPU** (or better)

3. Run cells top to bottom. Every pipeline stage resumes where it stopped, so re-running a cell after an interruption continues instead of starting over.

---

## Where Data Lives

| Location | Contents | Why |
|---|---|---|
| Colab local disk — `/content/TerraSR/data/` | downloads, standardized scenes, patches, LR/HR pairs | fast for tens of thousands of small files; wiped when the runtime is recycled |
| Google Drive — `MyDrive/TerraSR-Colab/` | checkpoints, results, WorldCover tile cache, dataset archive | persists across sessions |

```
MyDrive/TerraSR-Colab/
├── checkpoints/        ← last.pth (every epoch) + best.pth per model
├── results/            ← evaluation report (.md / .docx)
├── cache/worldcover/   ← ESA WorldCover tiles (~110 MB each, downloaded once)
├── dataset_archives/
│   └── terrasr_dataset.tar   ← the finished dataset, one file (Section 6b)
└── logs/
```

The notebook links `/content/TerraSR/checkpoints` and `/content/TerraSR/data/cache` to Drive, so all relative paths in the research code resolve without config changes.

### Why the dataset is not kept on Drive

Earlier versions of this notebook stored every intermediate file on Drive. Patchify writes one GeoTIFF per 256×256 patch — the default subset alone is ~20,000–28,000 files — and writing that many small files one by one through the Drive mount took hours and led to disconnects. Local disk handles it in minutes. The finished dataset is then saved to Drive as **one archive**, which copies quickly and is restored in one step.

---

## Workflow

### First session — build the dataset
1. **Sections 0–4:** GPU check, Drive mount, clone + install, data paths, import check
2. **Section 4b:** reports that there is nothing to restore yet
3. **Section 5:** download a subset to local disk (scenes left on Drive by an earlier notebook version are reused, not re-downloaded)
4. **Section 6:** pipeline stages 2–6 (resumable)
5. **Section 6b:** save the dataset archive to Drive

### Later sessions — skip the pipeline
1. **Sections 0–4**
2. **Section 4b:** restores the dataset archive from Drive to local disk
3. Continue at **Section 7** (verify) or **Section 11** (training)

### Training and evaluation
- **Sections 8–10:** single-batch test, 5-epoch smoke run, resume test
- **Section 11:** full training of all four models (SRCNN, SRGAN, SwinIR, TerraSR)
- **Section 12:** evaluation — report written to `MyDrive/TerraSR-Colab/results/`
- **Section 13:** download the report

---

## Resuming After a Disconnect

Colab disconnects idle browser tabs and caps session length (limits vary by tier and change over time). Everything is built to resume:

| Interrupted during | What to do |
|---|---|
| Section 6 (pipeline), runtime still alive | Re-run Sections 0–4, then the stage cell that was running. It continues from where it stopped. |
| Section 6, runtime recycled (local disk empty) | Re-run Sections 0–6. Scenes re-download from AWS; nothing half-finished is ever reused because every file is written atomically. |
| Section 11 (training) | Re-run Sections 0–4, **Section 4b** (restore dataset), then Section 11. Each model resumes from its last completed epoch; finished models are skipped. |

How each stage resumes:
- **standardize** skips scenes already converted
- **patchify** skips finished scenes (recorded in `data/patches/_scene_manifests/`) and patches already on disk
- **WorldCover labelling** and **LR/HR pair generation** save progress every 500 patches and skip finished ones
- **training** reloads `last.pth` (model, optimizer, epoch, RNG state) from Drive

Pass `--fresh` to any of these scripts to ignore previous progress.

---

## Scaling Up the Dataset

Edit the download cell in **Section 5**:

```python
SPACENET_AOIS  = ["AOI_2_Vegas", "AOI_5_Khartoum", "AOI_3_Paris"]
SPACENET_MAX_FILES = None   # None = every scene in the AOI

MAXAR_EVENTS   = ["Brazil-Flooding-May24", "Belize-Wildfires-June24"]
MAXAR_MAX_FILES = None
```

Then re-run Sections 5, 6 and 6b. Already-processed files are skipped.

**Size guide:** each SpaceNet PAN scene is 16384×16384 px (~537 MB) and yields about 4,000 patches. Colab's local disk is smaller than a workstation's (check the free space Section 3 prints) — a few AOIs fit comfortably; the full 300–400 GB research dataset does not. For Colab, tens of thousands of patches is a solid training set.

Whole-AOI mosaic files (e.g. `AOI_2_Vegas_PAN_COG.tif`) are skipped automatically: they are built from the same scenes as the individual strips, so downloading both would put identical ground into the training and test sets.

---

## Batch Size Guide

| Colab GPU | Recommended batch_size |
|---|---|
| T4 (15 GB) | 8 |
| L4 (22 GB) | 12–16 |
| A100 (40 GB) | 16 |

Section 11 picks a batch size from the detected VRAM; edit `BATCH_SIZE` there if you hit out-of-memory errors.

---

## Relationship to the Research Codebase

The notebook is a thin orchestration layer over the research code. Models, training scripts, configs and evaluation are shared unchanged; the data pipeline scripts it calls are the same ones used on the RESOLVE workstation, and their resume support works identically there. Nothing in `colab/` is needed to run on a workstation.
