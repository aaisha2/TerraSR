"""Shared helpers for stage 4 labeling scripts."""
import math
from pathlib import Path

import numpy as np
import rasterio
import requests
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

WORLDCOVER_CLASS_NAMES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland",
    50: "Built-up", 60: "Bare/sparse vegetation", 70: "Snow and ice",
    80: "Permanent water bodies", 90: "Herbaceous wetland", 95: "Mangroves",
    100: "Moss and lichen",
}


def worldcover_tile_name(lon: float, lat: float, tile_grid_deg: int = 3) -> str:
    """ESA WorldCover tiles are named by their SW corner, floored to the
    tile grid (verified 2026-07-11 against the real bucket, e.g. a point at
    lon=-51.57, lat=-29.40 -> tile S30W054)."""
    lon_floor = int(math.floor(lon / tile_grid_deg) * tile_grid_deg)
    lat_floor = int(math.floor(lat / tile_grid_deg) * tile_grid_deg)
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"{ns}{abs(lat_floor):02d}{ew}{abs(lon_floor):03d}"


def worldcover_url(tile: str, cfg: dict) -> str:
    filename = cfg["filename_template"].format(tile=tile)
    return f"{cfg['base_url']}/{filename}"


def get_local_worldcover_tile(lon: float, lat: float, cfg: dict) -> Path:
    """Return a local path to the WorldCover tile covering (lon, lat),
    downloading it once (cached by tile name) if not already present.

    GDAL's /vsicurl/ streaming hit intermittent connection resets against
    this bucket when tested here (2026-07-11), independent of URL style,
    while plain `requests` downloads were reliable -- so tiles are fetched
    whole with `requests` and read locally, trading a one-time ~110MB/tile
    download for robustness. Many patches from the same AOI share a tile,
    so this is a one-time cost per 3x3-degree cell, not per patch."""
    tile = worldcover_tile_name(lon, lat, cfg["tile_grid_deg"])
    cache_dir = Path(cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{tile}.tif"

    if not dest.exists():
        url = worldcover_url(tile, cfg)
        tmp = dest.with_suffix(".tif.part")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        tmp.rename(dest)

    return dest


def read_worldcover_window_for_patch(patch_path, cfg: dict) -> np.ndarray:
    """Read the ESA WorldCover pixels covering a patch's footprint, from a
    locally cached tile (see get_local_worldcover_tile). Returns the raw
    class-code array at WorldCover's native 10m resolution -- deliberately
    not resampled onto the patch's fine pixel grid, since upsampling a
    coarse label doesn't add information for a majority-vote zonal stat."""
    with rasterio.open(patch_path) as patch_ds:
        lonlat_bounds = transform_bounds(patch_ds.crs, "EPSG:4326", *patch_ds.bounds)

    centroid_lon = (lonlat_bounds[0] + lonlat_bounds[2]) / 2
    centroid_lat = (lonlat_bounds[1] + lonlat_bounds[3]) / 2
    tile_path = get_local_worldcover_tile(centroid_lon, centroid_lat, cfg["worldcover_source"])

    with rasterio.open(tile_path) as wc_ds:
        window = from_bounds(*lonlat_bounds, transform=wc_ds.transform)
        data = wc_ds.read(1, window=window)
    return data


def class_histogram(arr: np.ndarray) -> dict:
    values, counts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}
