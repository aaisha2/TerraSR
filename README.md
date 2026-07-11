# TerraSR

Multi-terrain satellite image super-resolution (SwinIR, terrain-embedding +
terrain-aware loss). See the full build plan artifact for the complete
pipeline design, dataset decisions, and paper references.

## Status

| Stage | Status |
|---|---|
| 0 — config | done (`configs/degradation.yaml`) |
| 1 — download | not started |
| 2 — standardize | not started |
| 3 — patchify | not started |
| 4 — terrain labeling | not started |
| 5 — degrade (LR/HR pairs) | **working**, validated on synthetic test patches only |
| 6 — package + split | not started |
| 7 — baselines (SRCNN/SRGAN/SwinIR) | not started |
| 8 — TerraSR model | not started |
| 9 — evaluation | not started |
| 10 — web app | not started |

## Stage 5 — degradation pipeline (current)

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
