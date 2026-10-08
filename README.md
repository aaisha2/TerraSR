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

## Google Colab

No workstation? [`colab/TerraSR_Colab.ipynb`](colab/TerraSR_Colab.ipynb) runs
the same pipeline and training on a Colab GPU — see
[colab/README.md](colab/README.md). Data is processed on Colab's local disk
and only checkpoints, results and a single dataset archive are kept on Drive.

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
| 4 — terrain labeling | **working** — WorldCover zonal stats + DEM slope (makes Mountain obtainable) + purity-thresholded assignment + label-quality audit with a manual-QC sample |
| 5 — degrade (LR/HR pairs) | **working** — 16-bit GeoTIFF output (lossless HR copy + georeferenced LR); parameters calibrated to sensor MTF/SNR and validated by `validate_degradation.py` (7/7) |
| 6 — package + split | **working** — unified manifest join + spatial-block split (blocks of ground, not scene names) + leakage audit proving no shared ground |
| 7 — baselines (SRCNN/SRGAN/SwinIR) | **working** — manifest-driven Dataset + all 3 models + model-agnostic trainer, smoke-tested end-to-end on CPU |
| 8 — TerraSR model | **working** — SwinIR + terrain embedding (FiLM) + terrain-aware loss, plus the 7-variant attribution ablation incl. the shuffled-label control |
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

With `inline_filter: true` (the default in `configs/patchify.yaml`),
`tile_extractor.py` skips mostly-nodata / blank grid cells before writing
them, using the same thresholds as `patch_filter.py` — on a full Maxar tile
that avoids writing 965 of 1,849 cells, and the final kept set is identical.

## Resuming interrupted runs

Every data stage can be stopped at any point and re-run with the same
command to continue — useful on Colab, and after a crash or power cut:

| Stage | What is skipped on re-run |
|---|---|
| 2 standardize | scenes already converted |
| 3 patchify | finished scenes (`<out>/_scene_manifests/`) and patches already on disk |
| 4 WorldCover labelling | patches already labelled (progress saved every 500) |
| 5 LR/HR pairs | pairs already written **whose degradation config still matches** (progress saved every 500) |
| 7–8 training | resumes from `last.pth` (the startup banner reports from where) |

All outputs are written to a temporary name and renamed when complete, so an
interrupted write never leaves a truncated file that a resume would treat as
finished. Each script takes `--fresh` to discard previous progress. Stage 2
also streams scenes in row strips, so memory stays bounded for scenes of any
size (the whole-scene reads it replaced needed 6+ GB for a SpaceNet mosaic).
When `configs/degradation.yaml` sets a `seed`, each LR/HR pair uses its own
RNG derived from the seed and patch id, so a resumed stage 5 produces exactly
the same pairs as an uninterrupted run.

Stage 5 also stamps every pair with a short fingerprint of the degradation
config that produced it, and reuses a pair on resume only if that fingerprint
still matches. Without this, resume cannot tell *"already done"* from *"done
differently"*: on Colab, 21,791 pairs built with the pre-2026-10 (over-strong)
config were reused unchanged after the config was corrected, and only
`validate_degradation.py` noticed. A config change now regenerates the affected
pairs instead of silently keeping them, so one dataset can never mix
degradation parameters.

## Stage 4 — terrain labeling

```bash
# 4a: per-patch ESA WorldCover class histogram (configs/terrain_classes.yaml)
python data_pipeline/04_labeling/worldcover_zonal_stats.py \
    --manifest out/patches/patch_manifest_filtered.json --only-kept

# 4a-bis: DEM slope + local relief - the only way the Mountain class can be
#         assigned (WorldCover has no landform classes)
python data_pipeline/04_labeling/dem_slope_stats.py \
    --manifest out/patches/patch_manifest_zonal.json --only-kept

# 4b: terrain label + purity/margin/confidence from the histogram and slope
python data_pipeline/04_labeling/assign_dominant_terrain.py \
    --manifest out/patches/patch_manifest_zonal.json

# 4c: label quality audit + a stratified sample for manual verification
python data_pipeline/04_labeling/label_quality_report.py \
    --manifest out/patches/patch_manifest_labeled.json \
    --report out/patches/label_quality.md
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

### Label quality

The labels are automated, so their quality is a property of the data and the
thresholds, not something to assume. Three defects were fixed in 2026-10, all
of which produced labels that looked clean in the manifest:

- **Nodata could win the majority vote.** Unmapped WorldCover codes (0 = no
  data, present on every coastal and scene-edge patch) were bucketed as
  `Unmapped` and voted like a real class, so a mostly-nodata patch was labelled
  `Unmapped` - a string absent from `terrain_index`, which the Dataset then
  silently mapped to the unknown embedding. Unmapped pixels are now excluded
  from the vote, purity is computed over mapped pixels only, and a patch with
  less than `purity.min_mapped_fraction` mapped is left unlabelled with a
  recorded reason.
- **No purity threshold.** `min_purity_to_keep_label` was `0.0`, so a 34%
  forest / 33% urban patch was labelled Forest as confidently as a 99% forest
  patch, and the terrain conditioning and per-terrain loss trained on that
  noise. It is now `0.6`, with the margin over the runner-up recorded per patch.
- **Mountain was unobtainable.** ESA WorldCover is a land-cover product with no
  landform classes, so Mountain came only from `scene_terrain_overrides` - and
  that dict was empty. Terrain index 4 had an embedding row in the stage 8 model
  and a weight in the terrain-aware loss that no training sample ever touched.
  It now comes from **Copernicus DEM GLO-30** slope and local relief (public,
  anonymous, 1-degree COG tiles, cached like the WorldCover tiles). Verified
  against the Grand Canyon tile: inner gorge 26.6 deg mean slope / 92 m relief
  and the rim 41.9 deg / 263 m are Mountain; the Kaibab and Coconino plateaus at
  8.8 and 10.5 deg are not.

Two land-cover mappings marked "approximation" were also resolved: Shrubland
was mapped to Bare Land/Desert (vegetated ground labelled as sand) and is now
Grassland/Wetland; Snow and ice was also mapped to Bare Land/Desert, which is
radiometrically its opposite, and is now left **unmapped**, so snow-dominated
patches go unlabelled rather than mislabelled.

Mountain is a landform while the other six classes are land cover, so they are
not mutually exclusive - `mountain.precedence` sets the convention, and each
patch records which rule applied.

`label_quality_report.py` reports support per class (naming any class with zero
patches, so the write-up cannot call it a terrain type), purity and margin
distributions, why patches went unlabelled, a purity-threshold sweep, DEM
coverage, and the WorldCover-2021 vs imagery-date gap. It also writes
`label_qc_sample.csv` - N patches per class with an empty `human_verdict`
column - so a **human-verified** label accuracy can be quoted. The automated
statistics describe how *decisive* the vote was, which is not the same thing as
whether it was *right*.

Because stage 4 now abstains on mixed patches, stage 6a **keeps** unlabelled
patches by default (`--drop-unlabeled` restores the old behaviour): dropping
them would delete a large share of the dataset and bias the rest towards pure
single-terrain scenes, when only the conditioning needs a label and the model
already has an unknown-terrain row.

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

### Ablation — what is the gain actually from?

TerraSR changes **two** things at once relative to SwinIR: it adds FiLM
conditioning (extra parameters) and it swaps L1 for a composite terrain-weighted
loss. So comparing only `terrasr` against `swinir + L1` **cannot attribute a
gain to terrain** — the identical number would appear if the gain came from the
extra capacity, or from the composite loss shape with terrain contributing
nothing at all. Without the grid below, the project's central claim is
unsupported whatever the headline PSNR says.

`train_terrasr.py` therefore takes three independent axes — `model.name` /
`model.conditioned`, `loss.mode`, and `data.terrain_label_mode` — and
`configs/ablation.yaml` defines the variants:

| variant | model | loss | labels | isolates |
|---|---|---|---|---|
| `swinir_l1` | SwinIR | L1 | — | the shared baseline |
| `swinir_uniform_loss` | SwinIR | composite, identical weights per terrain | real | the loss *shape* |
| `swinir_terrainloss` | SwinIR | terrain-aware | real | the loss half |
| `terrasr_l1` | TerraSR | L1 | real | the architecture half |
| `terrasr_full` | TerraSR | terrain-aware | real | the full method |
| `terrasr_shuffled_labels` | TerraSR | terrain-aware | **permuted** | **the control** |
| `terrasr_unknown_labels` | TerraSR | terrain-aware | all unknown | conditioning with no signal |

```bash
python training/run_ablation.py                     # resumable, variant by variant
python evaluation/eval_ablation.py --runs-root checkpoints/ablation \
    --test-csv data/dataset/test.csv --report data/dataset/ablation.md
```

`terrasr_shuffled_labels` is the one that decides the claim: identical
architecture, identical parameter count, identical loss family — the only
difference is that each patch is handed *another* patch's terrain label. If
`terrasr_full` does not beat it by more than seed noise, the conditioning is not
using terrain information, and whatever gain the full model shows over the
baseline comes from capacity and the loss. Each variant is also evaluated under
the label condition it was *trained* with, since a shuffled-label control
measured with real labels is not the model that was trained.

`eval_ablation.py` builds the attribution chain (`headline claim`, `terrain
information`, `conditioning alone`, `loss alone`, `per-terrain weighting`),
reports parameter counts beside each difference — a gain that tracks parameter
count rather than terrain is not a terrain result — and marks any difference
smaller than the noise floor as not interpretable. With one seed the floor is an
assumption from the config and the report says so; with several seeds it uses
the **measured** spread instead. Start at `seeds: [42]` to get the grid running,
then extend before quoting a sub-0.1 dB effect.

Verified end to end on the UC Merced fixtures: all 7 variants train, resume and
evaluate, with the controls param-matched at +0 and conditioning adding exactly
+7,960 parameters.

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

Both trainers report progress the same way: a startup banner (model, data,
where it is resuming from, how much is left), a timestamped batch line every
`train.log_every` batches with an it/s rate and epoch ETA, and a per-epoch line
with loss, validation PSNR and a run ETA. Each completed epoch is appended to
`<out_dir>/training_log.csv`, so progress survives a lost console:

```bash
python training/training_status.py --dir checkpoints --tail 10
```

## Stage 6 — package + split

```bash
# 6a: join stage 4 (terrain labels) + stage 5 (LR/HR pairs) into one manifest
python data_pipeline/06_package/build_manifest.py \
    --labeled out/patches/patch_manifest_labeled.json \
    --degradation out/pairs/degradation_manifest.json \
    --out-dir out/dataset

# 6b: geographic-block train/val/test split (blocks of GROUND, not scene names)
python data_pipeline/06_package/split_train_val_test.py \
    --manifest out/dataset/dataset_manifest.parquet

# 6c: prove the split shares no ground between train and test
python data_pipeline/06_package/audit_split_leakage.py \
    --manifest out/dataset/dataset_manifest_split.csv \
    --report out/dataset/split_leakage_audit.md --strict

# 6d: sanity report — per-terrain counts, floor check
python data_pipeline/06_package/dataset_stats.py \
    --manifest out/dataset/dataset_manifest_split.parquet
```

If a dataset was built before stage 3 recorded patch geography, backfill it
instead of rebuilding (reads the geotransform off each patch GeoTIFF):

```bash
python data_pipeline/06_package/backfill_patch_geo.py \
    --manifest out/dataset/dataset_manifest.csv
```

The unified manifest (`dataset_manifest_split.parquet`/`.csv`, plus
`train.csv`/`val.csv`/`test.csv`) is the single source of truth for stages
7–9: one row per patch with `hr_path`, `lr_path`, `terrain_label`,
`terrain_purity`, `pseudo_pan`, `source_scene`, and `split`.

Split is by **geographic block**: patch centroids are quantised to a
`block_size_km` grid (longitude step widened by 1/cos(lat) so blocks stay
roughly square), whole blocks are assigned to one split, and patches within
`buffer_m` of a differently-assigned block are dropped entirely
(`split='excluded_buffer'`). Assignment uses a normalized-deficit greedy that
converges to the configured 80/10/10 with many blocks and still spreads blocks
across splits when there are few.

**Why blocks and not scenes.** The split used to treat `source_scene` as the
atomic unit. That stops adjacent-patch leakage *inside* a scene but not
same-location leakage *between* scenes — and five of this project's sources
image the same ground under different scene names:

| Source | Same ground appears as |
|---|---|
| CORE3D | ~154 scenes over 5 sites, i.e. tens of repeat WorldView collects of one city |
| Maxar events | pre-event and post-event acquisitions of the same quadkeys |
| `maxar_visual` | the *same scenes* as `maxar`, just the `visual` asset instead of `pan_analytic` |
| `naip_pc` | its AOIs are deliberately chosen to overlap the CORE3D sites |
| SpaceNet | strips within an AOI overlap at the edges |

Measured on a fixture reproducing those routes, the scene-name split put **33%
of val and 33% of test patches within 0 m of a training patch** — the same
ground, in both sets, so the reported test PSNR/SSIM was partly a memorisation
score. The block split passes the same audit with a 468 m minimum separation.

`audit_split_leakage.py` is what verifies this rather than assuming it: for
every held-out patch it measures the distance to the nearest *training* patch
(KD-tree in a local metric projection) and fails if any is closer than a patch
footprint. It also reports the routes a scene-name split cannot see — locations
appearing in two splits, repeat coverage straddling splits, identical geometry
twice. `--strict` makes it gate the pipeline.

One consequence worth stating in the write-up: a scene now *may* contribute to
several splits, because a scene covers many blocks of ground and blocks are the
unit. That is correct — those are different locations. What must never happen is
one location in two splits, and that is what the audit checks.

Every source must go through **one** manifest and **one** split. Splitting
true-PAN and pseudo-PAN separately re-introduces the leakage, because the block
assignment would be independent; Experiment 1 vs 2 is selected at training time
with `data.pan_filter`.

## Stage 5 — degradation pipeline

Single-order pipeline confirmed with supervisor 2026-07-11: randomized
blur (Gaussian / anisotropic Gaussian / MTF-elliptical) -> randomized
downsample (nearest/area/bicubic) -> Poisson+Gaussian noise -> optional JPEG
recompression. All parameters live in `configs/degradation.yaml`.

### Calibration and validation

The LR/HR pairs *define the task*, so the degradation is calibrated against the
sensor literature and re-measured from the built dataset. Two errors had gone
unnoticed because nothing checked, and they pulled in opposite directions:

- **A unit bug in the MTF kernel family.** A sigma derived from a Nyquist-MTF
  target is in *LR* pixels, but the blur runs on the HR grid before
  downsampling, so it must be scaled by `scale_factor`. It was used unscaled,
  so that family blurred ~2× too *little* (realized MTF 0.73-0.80 against its
  own 0.20-0.35 target).
- **The Gaussian families were tuned by eye**, far past the MTF the config
  cites: sigma 0.6-2.4 px HR == MTF 0.64 down to 0.0008 at the LR Nyquist
  frequency. Measured medians were 0.049 and 0.063 against the MTF family's
  0.265 — the three families disagreed with each other by about 15×.

Poisson noise also reached a shot SNR of **3.5** at full signal where real
WorldView-class PAN is ≥100, and JPEG was on by default — which models a
delivery artifact that 16-bit archive PAN never has, and silently quantises the
LR to 8 bits inside the stage whose job is preserving 16-bit depth.

Net effect, measured on the UC Merced patches: the LR images sat **16.3 dB**
from an ideal 2× resample with high-pass noise **15.8× the scene's own
texture**. Models were being trained to undo synthetic noise and blur, so any
PSNR gain over bicubic was inflated by that rather than by super-resolution.

Every range is now stated in, or derived from, a physical quantity with the
derivation in the config, and `degrade()` logs the full per-patch parameter set
so a built dataset is auditable after the fact:

```bash
# config only (fast, no dataset needed)
python data_pipeline/05_degrade/validate_degradation.py

# full audit of a built dataset; non-zero exit if a check fails
python data_pipeline/05_degrade/validate_degradation.py \
    --manifest out/pairs/degradation_manifest.json --strict \
    --report out/pairs/degradation_validation.md
```

It measures kernel MTF by FFT (no Gaussian assumption), summarises what the
dataset recorded, and re-derives LR-vs-ideal-LR PSNR, noise level and spectrum
from the written pairs. The corrected config scores **7/7**; the original
scores **1/7**, and restoring the unit bug is caught by the
family-self-consistency check.

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
