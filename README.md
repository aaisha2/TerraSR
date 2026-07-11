# TerraSR

Multi-terrain satellite image super-resolution (SwinIR, terrain-embedding +
terrain-aware loss). See the full build plan artifact for the complete
pipeline design, dataset decisions, and paper references.

## Status

| Stage | Status |
|---|---|
| 0 — config | done (`configs/degradation.yaml`, `configs/datasets.yaml`) |
| 1 — download | **working** — SpaceNet (PAN) + Maxar Open Data (pan_analytic), verified against real buckets |
| 2 — standardize | not started |
| 3 — patchify | not started |
| 4 — terrain labeling | not started |
| 5 — degrade (LR/HR pairs) | **working**, validated on synthetic test patches only |
| 6 — package + split | not started |
| 7 — baselines (SRCNN/SRGAN/SwinIR) | not started |
| 8 — TerraSR model | not started |
| 9 — evaluation | not started |
| 10 — web app | not started |

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

## Stage 5 — degradation pipeline

Single-order pipeline confirmed with supervisor 2026-07-11: randomized
blur (Gaussian / anisotropic Gaussian / MTF-elliptical) -> randomized
downsample (nearest/area/bicubic) -> Poisson+Gaussian noise -> JPEG
recompression (quality 70-95). All parameters live in
`configs/degradation.yaml`.

```bash
pip install -r requirements.txt

# generate LR/HR pairs from a folder of HR patches (PNG/JPG/TIF for now)
python data_pipeline/05_degrade/make_lr_hr_pairs.py \
    --hr-dir tests/sample_images --out-dir tests/out/pairs

# compare synthetic LR against real satellite LR reference chips
# (required by supervisor before locking dataset-wide parameters)
python data_pipeline/05_degrade/validate_against_real_lr.py \
    --synthetic-dir tests/out/pairs/lr --real-dir <path-to-real-LR-chips>
```

**`tests/sample_images/` are synthetic placeholder patches** (procedurally
generated grid/building/texture pattern), not real satellite imagery —
they exist only to smoke-test the pipeline mechanics before SpaceNet/Maxar
data is available. The `validate_against_real_lr.py` run against them is
*not* a real validation; it needs actual reference LR chips from a
comparable sensor once stage 1 (download) is in place.

## Requirements

`pip install -r requirements.txt`. `rasterio`/GDAL gets exercised starting
stage 2 (GeoTIFF standardization) — not needed for stage 5 alone.
