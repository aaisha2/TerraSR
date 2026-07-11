# TerraSR

Multi-terrain satellite image super-resolution (SwinIR, terrain-embedding +
terrain-aware loss). See the full build plan artifact for the complete
pipeline design, dataset decisions, and paper references.

## Status

| Stage | Status |
|---|---|
| 0 — config | done (`configs/degradation.yaml`, `configs/datasets.yaml`) |
| 1 — download | **working** — SpaceNet (PAN) + Maxar Open Data (pan_analytic), verified against real buckets |
| 2 — standardize | **working** — PAN/pseudo-PAN extraction + CRS/dtype normalization, tested on a real Maxar crop and a synthetic geographic RGB fixture |
| 3 — patchify | **working** — fixed-grid patch extraction + nodata/blank/saturation filtering, tested on a real Maxar crop straddling a nodata boundary |
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
