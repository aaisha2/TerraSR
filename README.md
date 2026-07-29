# TerraSR

Multi-terrain satellite image super-resolution (SwinIR, terrain-embedding +
terrain-aware loss). See the full build plan artifact for the complete
pipeline design, dataset decisions, and paper references.

## Full run (RESOLVE workstation)

The entire data build (stages 1–6) runs in one command via the orchestrator,
which drives every stage on the real `data/` tree defined in
`configs/pipeline.yaml`:

```bash
python run_pipeline.py --list-only     # preview the complete download size
python run_pipeline.py                 # full data build: download -> dataset/
python run_pipeline.py --with-training # + train baselines/TerraSR + evaluate
```

**See [RESOLVE.md](RESOLVE.md) for the complete runbook** — environment setup
(CUDA torch), disk budget (~300–400 GB), the complete download, and
training/eval. Downloaded imagery and all intermediates live under `data/`
(gitignored); only code and configs are versioned, so RESOLVE just pulls the
repo and builds `data/` locally.

## Viewing the imagery (16-bit GeoTIFFs)

Stages 2–5 write 16-bit single-band GeoTIFFs, which Windows Photos can't open —
and viewers that can usually render them near-black, because PAN data occupies
only a slice of the 16-bit range. `tools/preview.py` applies a percentile
contrast stretch (what a GIS viewer does) so you can actually see them:

```bash
# browse any stage's output: PNGs on disk + a self-contained contact sheet
python tools/preview.py --input data/standardized/maxar --out-dir previews/std
python tools/preview.py --input data/patches --html previews/patches.html --limit 40

# tune configs/degradation.yaml: HR vs LR side by side, plus matched zoom crops
# where blur / noise / aliasing are actually visible, annotated with the params
python tools/preview.py --pairs data/pairs --html previews/degradation.html --limit 12
```

The HTML pages are self-contained (images embedded) — just double-click to open
in any browser. Stretch is configurable so you can compare like-for-like:
`--stretch percentile|minmax|none` (`none` shows the raw mapping a naive viewer
would give). `previews/` is gitignored.

## Status

| Stage | Status |
|---|---|
| 0 — config | done (`configs/degradation.yaml`, `configs/datasets.yaml`) |
| 1 — download | **working** — SpaceNet (PAN) + Maxar Open Data (pan_analytic), verified against real buckets |
| 2 — standardize | **working** — PAN/pseudo-PAN extraction + CRS/dtype normalization, tested on a real Maxar crop and a synthetic geographic RGB fixture |
| 3 — patchify | **working** — fixed-grid patch extraction + nodata/blank/saturation filtering, tested on a real Maxar crop straddling a nodata boundary |
| 4 — terrain labeling | **working** — ESA WorldCover zonal stats + dominant-terrain assignment, tested end-to-end on real patches from the stage 3 output |
| 5 — degrade (LR/HR pairs) | **working** — 16-bit GeoTIFF output (lossless HR copy + georeferenced LR), manifest-drivable; legacy 8-bit PNG path retained for smoke tests |
| 6 — package + split | **working** — unified manifest join + geographic-block split + dataset stats, tested end-to-end on 3 real Maxar-derived scenes |
| 7 — baselines (SRCNN/SRGAN/SwinIR) | **working** — manifest-driven Dataset + all 3 models + model-agnostic trainer, smoke-tested end-to-end on CPU |
| 8 — TerraSR model | **working** — SwinIR + terrain embedding (FiLM) + terrain-aware loss, smoke-tested end-to-end on CPU |
| 9 — evaluation | **working** — overall + per-terrain PSNR/SSIM (tested); downstream-detection harness (proxy metric, documented) |
| 10 — web app | **working** — FastAPI backend (loads a trained checkpoint) + build-free browser UI (upload, terrain select, before/after slider), verified end-to-end |

## Stage 1 — download

Both sources are public and need no AWS account/credentials — verified
directly against the live buckets on 2026-07-11:

- **SpaceNet** (`s3://spacenet-dataset/AOIs/<AOI>/PAN/*.TIF`) — anonymous
  unsigned S3 access works despite older docs describing the bucket as
  requester-pays; that's not enforced on the current layout.
- **Maxar Open Data** (STAC catalog at `maxar-opendata.s3.amazonaws.com`) —
  walks `events/catalog.json` -> event collection -> acquisition
  collections -> STAC items -> `assets.pan_analytic.href`, plain HTTPS GET.

Both scripts are idempotent (skip files already on disk with a matching
size) and support `--list-only` to preview without downloading.

```bash
python data_pipeline/01_download/download_spacenet.py --list-only --aoi AOI_2_Vegas --max-files 5
python data_pipeline/01_download/download_maxar.py --list-only --event Brazil-Flooding-May24

# real download, capped for a smoke test
python data_pipeline/01_download/download_maxar.py --event Brazil-Flooding-May24 --max-files 1
```

Downloaded imagery lands under `data/raw/` (gitignored — this is real
multi-hundred-MB-per-scene satellite data, never committed).

## Stage 2 — standardization

Two chained scripts per patch/scene:

```bash
# 2a: pull out the PAN (or pseudo-PAN) band, tagging pseudo_pan true/false
python data_pipeline/02_standardize/extract_pan_band.py \
    --in raw_scene.tif --out extracted.tif --mode true_pan   # or rgb_to_pseudo_pan

# 2b: normalize CRS (reprojects to local UTM only if source is geographic —
# already-projected scenes like SpaceNet/Maxar pass through untouched) and
# dtype (-> uint16), write compressed/tiled GeoTIFF
python data_pipeline/02_standardize/to_geotiff.py --in extracted.tif --out standardized.tif

# batch: standardize a whole source directory in one call (chains 2a+2b per
# scene with the right PAN mode) — this is what the orchestrator calls
python data_pipeline/02_standardize/standardize_scenes.py \
    --in-dir data/raw/spacenet --out-dir data/standardized/spacenet --mode true_pan --recursive
```

Tested against a real cropped window of the downloaded Maxar PAN tile
(true-PAN passthrough, no unnecessary reprojection since it's already in
UTM) and a synthetic RGB fixture in geographic coordinates (exercises the
luminance pseudo-PAN conversion and the UTM auto-reprojection path).
Output tags (`pseudo_pan`, `orig_crs`, `orig_dtype`, `reprojected_to_utm`)
carry through so later stages — and any ablation — can tell true PAN from
pseudo-PAN without re-deriving it.

## Stage 3 — patchify

```bash
# 3a: cut a standardized scene into a fixed 256x256 grid (configs/patchify.yaml)
python data_pipeline/03_patchify/tile_extractor.py \
    --in-scene standardized.tif --out-dir out/patches
# or batch a whole directory of standardized scenes:
#   --in-dir out/standardized_scenes --out-dir out/patches

# 3b: drop nodata/blank/saturated patches (non-destructive by default —
# writes patch_manifest_filtered.json with keep/drop + reasons per patch;
# add --move-rejected to physically relocate dropped patches)
python data_pipeline/03_patchify/patch_filter.py --manifest out/patches/patch_manifest.json
```

Tested on a real 1600x1600 Maxar crop deliberately straddling a nodata
boundary in the source scene: extraction produced the expected 6x6 grid of
256px patches, and the filter correctly dropped every patch that was fully
or partially nodata (`nodata_fraction`/`std_dev` thresholds) while keeping
the patches with real terrain content — confirmed visually and by pixel
stats (dropped patch: min=max=0; kept patch: real DN range, visible
field/path texture).

Each patch keeps its source scene's tags (`pseudo_pan`, `orig_crs`, etc.)
plus `source_scene`/`row`/`col`/`tile_id`, which stage 6's geographic
split depends on to keep all patches from one scene in the same split.

## Stage 4 — terrain labeling

```bash
# 4a: per-patch ESA WorldCover class histogram (configs/terrain_classes.yaml)
python data_pipeline/04_labeling/worldcover_zonal_stats.py \
    --manifest out/patches/patch_manifest_filtered.json --only-kept

# 4b: dominant terrain label + purity from the histogram
python data_pipeline/04_labeling/assign_dominant_terrain.py \
    --manifest out/patches/patch_manifest_zonal.json
```

WorldCover tiles are cached locally on first use rather than streamed via
GDAL's `/vsicurl/` — that streaming path hit intermittent connection resets
against the bucket when tested here (independent of URL style), while plain
`requests` downloads were reliable, so tiles are fetched whole once per
3°×3° cell and read locally afterward (many patches from one AOI share a
tile, so this is a one-time cost, not per-patch).

Tested end-to-end on the real kept patches from stage 3: the WorldCover
histogram for one patch (Forest, purity 0.85) was `{Tree cover: 165,
Grassland: 30}` — cross-checked against the patch's own visual preview
(field/forest/path texture), which matches.

**Known gap, flagged rather than silently patched over:** ESA WorldCover
is a land-cover product with no landform classes, so "Mountain" cannot be
derived from it per-pixel — Shrubland/Bare/Snow occur on mountains and on
flat drylands alike. Mountain is therefore only assigned via
`scene_terrain_overrides` in `configs/terrain_classes.yaml` (a source-scene
name match, e.g. tagging a whole mountainous Maxar event), not from the
pixel histogram. A DEM-slope-based per-pixel refinement (Copernicus
DEM/SRTM — already scoped as a reserve source in the build plan) would be
the correct long-term fix but isn't implemented yet.

## Stage 10 — web app

A FastAPI backend (`webapp/backend/app.py`) loads a trained checkpoint and
super-resolves an uploaded image; a self-contained browser UI
(`webapp/frontend/index.html`, no build step) handles upload, terrain
selection, and a draggable before/after comparison. Because TerraSR is
terrain-conditioned, the UI's terrain dropdown feeds the terrain embedding.

```bash
pip install -r webapp/requirements.txt
python webapp/backend/app.py            # http://127.0.0.1:8000
```

Verified end-to-end: `/api/health` reports the loaded model, `/api/infer`
returns a real 128²→256² terrain-conditioned result, and the UI renders the
comparison slider + metadata. If no checkpoint exists yet it falls back to
bicubic (clearly labelled) so it's demonstrable before the RESOLVE run. See
[webapp/README.md](webapp/README.md). Automatic terrain classification from the
LR image is noted there as future work.

## Stage 9 — evaluation

```bash
# 9a: overall PSNR/SSIM for any checkpoints + the bicubic floor
python evaluation/eval_psnr_ssim.py --test-csv data/dataset/test.csv --with-bicubic \
    --checkpoints checkpoints/swinir_baseline/best.pth checkpoints/terrasr/best.pth

# 9b: per-terrain PSNR/SSIM breakdown (the key deliverable) + per-terrain delta
python evaluation/eval_per_terrain.py --test-csv data/dataset/test.csv \
    --checkpoints checkpoints/swinir_baseline/best.pth checkpoints/terrasr/best.pth

# 9c: downstream-detection harness (proxy metric — see below)
python evaluation/eval_downstream_detection.py --test-csv data/dataset/test.csv \
    --checkpoint checkpoints/terrasr/best.pth
```

All three load any checkpoint (baseline or terrain-conditioned) and dispatch
the forward call correctly (`model(lr)` vs `model(lr, terrain_idx)`) from the
`model_name` saved in the checkpoint. Metrics use `skimage.metrics` (the
community-standard implementations) so numbers are comparable to the SR
literature. `eval_per_terrain.py` prints a per-terrain delta when exactly two
checkpoints are compared — this is where the terrain-aware gain shows up
terrain by terrain, not just in the average.

**Downstream detection is a documented proxy, not the final metric.** The
rigorous version needs ground-truth object boxes (e.g. SpaceNet building
footprints) and a detector fine-tuned on them, then reports mAP on LR vs SR vs
HR. That GT + fine-tuned detector isn't wired up yet, so the harness reports a
runnable stand-in: using a pretrained COCO detector as a fixed reference, how
well each SR output reproduces the detections the detector makes on the true
HR image (detection-consistency). Swap in the SpaceNet-fine-tuned detector and
GT boxes to get the final mAP number.

## Stage 8 — TerraSR model (the novel contribution)

Combines the two ideas from the proposal into one model, not seven:

- **Terrain embedding + FiLM** (`models/terrasr_swinir.py`): a learned
  embedding table holds one vector per terrain; each RSTB has its own
  projection from that embedding to per-channel `(gamma, beta)` that modulate
  the block's conv features (`f' = (1 + gamma) * f + beta`). The FiLM
  generator is **zero-initialised**, so at the start of training terrain
  conditioning is exactly the identity — it can only help, never disrupt the
  base network. Verified: at init, urban-vs-no-terrain output difference is
  exactly 0; after the FiLM weights move, urban vs forest outputs genuinely
  diverge.
- **Terrain-aware loss** (`models/losses/`): a shared L1 + SSIM base plus
  per-terrain edge and perceptual terms (`configs/terrain_aware_loss.yaml`) —
  urban up-weights the edge/gradient term, forest up-weights the
  perceptual/texture term, water/desert keep both low to avoid ringing. Each
  sample in a mixed-terrain batch is weighted by its own terrain. The VGG
  perceptual term is built lazily (only if some terrain asks for it), so runs
  that disable it never need the 528 MB VGG download.

```bash
python training/train_terrasr.py --config configs/train_terrasr.yaml
# quick offline run (no VGG download):
python training/train_terrasr.py --config configs/train_terrasr.yaml \
    --override train.epochs=5 loss.disable_perceptual=true
```

`train_terrasr.py` differs from the baseline trainer in exactly two places —
the model receives the per-sample terrain index (`model(lr, terrain_idx)`)
and the loss is the terrain-aware composite — so results stay directly
comparable to the baselines under matched conditions. Smoke-tested
end-to-end on CPU (perceptual disabled): trains, validates, checkpoints.

## Stage 7 — baselines (SRCNN / SRGAN / SwinIR)

The stage 6 manifest feeds a single manifest-driven PyTorch `Dataset`
(`terrasr_data/dataset.py`) that yields `(lr, hr, terrain_idx, meta)` for
every model — the baselines ignore `terrain_idx`, stage 8 uses it, so the
data path never changes between experiments. All three models are
single-channel (PAN) and upscale 2×:

- `models/srcnn_baseline.py` — SRCNN (CNN baseline)
- `models/srgan_baseline.py` — SRResNet generator + discriminator (GAN baseline)
- `models/swinir_baseline.py` — SwinIR, written in-house so stage 8 can inject
  FiLM terrain conditioning into the RSTB blocks

`training/train_baseline.py` is model-agnostic — the model is chosen by
config, so benchmarking three baselines under identical conditions is three
config files (or `--override model.name=...`).

```bash
python training/train_baseline.py --config configs/train_baseline.yaml
# swap model / tweak without editing the file:
python training/train_baseline.py --config configs/train_baseline.yaml \
    --override model.name=srcnn train.epochs=5
```

Smoke-tested end-to-end on CPU on the real GeoTIFF dataset: all three models
train (loss decreasing), validate (per-epoch PSNR), and save/reload
checkpoints. **The PSNR numbers from those runs are meaningless** (16 patches,
2–3 epochs) — they only prove the machinery; real training runs on the
RESOLVE GPU with the full dataset.

## Stage 6 — package + split

```bash
# 6a: join stage 4 (terrain labels) + stage 5 (LR/HR pairs) into one manifest
python data_pipeline/06_package/build_manifest.py \
    --labeled out/patches/patch_manifest_labeled.json \
    --degradation out/pairs/degradation_manifest.json \
    --out-dir out/dataset

# 6b: geographic-block train/val/test split (whole scenes, never per-patch)
python data_pipeline/06_package/split_train_val_test.py \
    --manifest out/dataset/dataset_manifest.parquet

# 6c: sanity report — per-terrain counts, floor check, leakage guard
python data_pipeline/06_package/dataset_stats.py \
    --manifest out/dataset/dataset_manifest_split.parquet
```

The unified manifest (`dataset_manifest_split.parquet`/`.csv`, plus
`train.csv`/`val.csv`/`test.csv`) is the single source of truth for stages
7–9: one row per patch with `hr_path`, `lr_path`, `terrain_label`,
`terrain_purity`, `pseudo_pan`, `source_scene`, and `split`.

Split is by **geographic block** — the whole source scene is the atomic
unit, never divided across splits, so near-duplicate neighbouring patches
can't leak train content into val/test. Assignment uses a normalized-deficit
greedy that converges to the configured 80/10/10 with many scenes and still
spreads scenes across splits when there are few. `dataset_stats.py` runs a
leakage guard confirming no scene spans multiple splits, and flags any
terrain below the per-class floor (surfacing the water/mountain scarcity
risk before training).

Tested end-to-end on 3 real scenes cropped from the downloaded Maxar tile,
run through stages 2→3→4→5 (GeoTIFF output): 48 patches joined cleanly, each
scene landed in a distinct split with zero leakage, and the stats report
correctly flagged Water/Mountain as absent.

## Stage 5 — degradation pipeline

Single-order pipeline confirmed with supervisor 2026-07-11: randomized
blur (Gaussian / anisotropic Gaussian / MTF-elliptical) -> randomized
downsample (nearest/area/bicubic) -> Poisson+Gaussian noise -> JPEG
recompression (quality 70-95). All parameters live in
`configs/degradation.yaml`.

**Output format** (`configs/degradation.yaml` -> `output.format`):

- `geotiff` (default, real pipeline) — preserves the 16-bit depth and
  georeferencing of the PAN patches. HR is written back as a **lossless
  copy** of the ground-truth patch (verified array-equal, same dtype/CRS/
  transform); LR is written as a 16-bit GeoTIFF with a geotransform scaled
  by the SR factor (verified: exactly 2× pixel size, same origin/CRS). For
  the degradation math the HR is normalized to `[0,1]` (per-patch max by
  default, so the `[0,1]`-relative noise params stay meaningful for PAN that
  only fills part of the uint16 range) and the LR is mapped back to the
  source dtype, so HR and LR share one radiometric scale.
- `png` (legacy) — 8-bit path used only for the synthetic smoke-test images.

```bash
pip install -r requirements.txt

# real pipeline: drive from the stage 4 labeled manifest so output aligns
# exactly with what stage 6 joins (only kept, labeled patches)
python data_pipeline/05_degrade/make_lr_hr_pairs.py \
    --manifest out/patches/patch_manifest_labeled.json --out-dir out/pairs --only-labeled

# or from a plain directory of HR images (used for the synthetic smoke test)
python data_pipeline/05_degrade/make_lr_hr_pairs.py \
    --hr-dir tests/sample_images --out-dir tests/out/pairs

# compare synthetic LR against real satellite LR reference chips
# (required by supervisor before locking dataset-wide parameters)
python data_pipeline/05_degrade/validate_against_real_lr.py \
    --synthetic-dir tests/out/pairs/lr --real-dir <path-to-real-LR-chips>
```

**`tests/sample_images/` are synthetic placeholder patches** (procedurally
generated grid/building/texture pattern), not real satellite imagery —
they exist only to smoke-test the pipeline mechanics. The
`validate_against_real_lr.py` run against them is *not* a real validation;
it needs actual reference LR chips from a comparable sensor.

## Requirements

`pip install -r requirements.txt`. `rasterio`/GDAL gets exercised starting
stage 2 (GeoTIFF standardization) — not needed for stage 5 alone.
