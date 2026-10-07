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


# Open WorldCover datasets, kept for the life of the process. Thousands of
# patches share each 3x3-degree tile; opening it once instead of once per
# patch matters most when the tile cache sits on a network mount (Google
# Drive on Colab), where every open/stat is a round trip.
_OPEN_TILES = {}


def _worldcover_dataset(lon: float, lat: float, cfg: dict):
    tile = worldcover_tile_name(lon, lat, cfg["tile_grid_deg"])
    ds = _OPEN_TILES.get(tile)
    if ds is None:
        ds = rasterio.open(get_local_worldcover_tile(lon, lat, cfg))
        _OPEN_TILES[tile] = ds
    return ds


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
    wc_ds = _worldcover_dataset(centroid_lon, centroid_lat, cfg["worldcover_source"])
    window = from_bounds(*lonlat_bounds, transform=wc_ds.transform)
    return wc_ds.read(1, window=window)


def class_histogram(arr: np.ndarray) -> dict:
    values, counts = np.unique(arr, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts)}


# --------------------------------------------------------------------------
# Copernicus DEM - the Mountain class
# --------------------------------------------------------------------------
# ESA WorldCover is a land-COVER product with no landform classes, so Mountain
# was previously reachable only through `scene_terrain_overrides` - a hand-kept
# dict of scene-name substrings that was empty, i.e. the class could never be
# assigned at all. Terrain index 4 (Mountain) therefore had an embedding row
# and a per-terrain loss weight that no training sample ever touched.
#
# Mountain is a property of the ground's shape, so it comes from a DEM.
# Copernicus DEM GLO-30 is public and anonymous over HTTPS (verified
# 2026-10-07: 1-degree COG tiles, ~39 MB each), and is cached locally exactly
# like the WorldCover tiles.
DEM_TILE_TEMPLATE = "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM"
DEM_BASE_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
METRES_PER_DEGREE = 111320.0


def dem_tile_name(lon: float, lat: float) -> str:
    """GLO-30 tiles are 1x1 degree, named by their SW corner."""
    lat_floor, lon_floor = int(math.floor(lat)), int(math.floor(lon))
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return DEM_TILE_TEMPLATE.format(ns=ns, lat=abs(lat_floor),
                                     ew=ew, lon=abs(lon_floor))


def get_local_dem_tile(lon: float, lat: float, cfg: dict) -> Path:
    """Local path to the DEM tile covering (lon, lat), downloaded once.

    Raises FileNotFoundError for a tile the dataset does not publish (GLO-30
    has no tiles over open ocean), so callers can treat 'no DEM here' as a
    normal outcome rather than a failure."""
    tile = dem_tile_name(lon, lat)
    cache_dir = Path(cfg.get("cache_dir", "data/cache/copernicus_dem"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{tile}.tif"
    missing_marker = cache_dir / f"{tile}.missing"

    if dest.exists():
        return dest
    if missing_marker.exists():
        raise FileNotFoundError(f"DEM tile {tile} is not published (cached result)")

    base = cfg.get("base_url", DEM_BASE_URL)
    url = f"{base}/{tile}/{tile}.tif"
    tmp = dest.with_suffix(".tif.part")
    with requests.get(url, stream=True, timeout=180) as r:
        if r.status_code in (403, 404):
            # remember it, so a coastal dataset doesn't re-request every patch
            missing_marker.write_text(f"{r.status_code} {url}\n")
            raise FileNotFoundError(f"DEM tile {tile} is not published ({r.status_code})")
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    tmp.replace(dest)
    return dest


_OPEN_DEM_TILES = {}


def _dem_dataset(lon: float, lat: float, cfg: dict):
    tile = dem_tile_name(lon, lat)
    if tile not in _OPEN_DEM_TILES:          # None caches "not published"
        try:
            _OPEN_DEM_TILES[tile] = rasterio.open(get_local_dem_tile(lon, lat, cfg))
        except FileNotFoundError:
            _OPEN_DEM_TILES[tile] = None
    ds = _OPEN_DEM_TILES[tile]
    if ds is None:
        raise FileNotFoundError(f"no DEM tile for lon={lon:.4f} lat={lat:.4f}")
    return ds


def patch_lonlat_bounds(patch_path):
    """A patch's footprint in WGS84 (left, bottom, right, top)."""
    with rasterio.open(patch_path) as ds:
        return transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)


def slope_stats(dem: np.ndarray, lat: float, dx_deg: float, dy_deg: float,
                 nodata=None) -> dict:
    """Slope and relief statistics for a DEM window.

    Spacing is converted from degrees to metres using the window's own
    transform (GLO-30 coarsens its longitude spacing above 50 degrees
    latitude, so this cannot be hard-coded) and cos(lat) for the longitude
    convergence. Slope is the gradient magnitude in degrees."""
    dem = np.asarray(dem, dtype=np.float64)
    if nodata is not None:
        dem = np.where(dem == nodata, np.nan, dem)
    if dem.size < 4 or np.all(np.isnan(dem)):
        return {"valid": False}

    dx_m = abs(dx_deg) * METRES_PER_DEGREE * max(math.cos(math.radians(lat)), 1e-6)
    dy_m = abs(dy_deg) * METRES_PER_DEGREE
    if dem.shape[0] < 2 or dem.shape[1] < 2 or dx_m <= 0 or dy_m <= 0:
        return {"valid": False}

    gy, gx = np.gradient(dem, dy_m, dx_m)
    slope_deg = np.degrees(np.arctan(np.hypot(gx, gy)))
    finite = slope_deg[np.isfinite(slope_deg)]
    elev = dem[np.isfinite(dem)]
    if finite.size == 0 or elev.size == 0:
        return {"valid": False}
    return {
        "valid": True,
        "slope_mean_deg": float(np.mean(finite)),
        "slope_p90_deg": float(np.percentile(finite, 90)),
        # local relief: the elevation span across the patch, the other standard
        # mountain criterion (a steep cliff face and a rolling hill differ here)
        "relief_m": float(np.percentile(elev, 95) - np.percentile(elev, 5)),
        "elevation_mean_m": float(np.mean(elev)),
        "dem_pixels": int(finite.size),
    }


def read_dem_stats_for_patch(patch_path, cfg: dict) -> dict:
    """Slope/relief statistics over a patch's footprint, from the cached
    Copernicus DEM. A patch is typically much smaller than one 30 m DEM
    pixel's neighbourhood, so the window is padded to at least `min_window`
    pixels - slope is a property of the surrounding landform, not of the
    77 m patch alone."""
    left, bottom, right, top = patch_lonlat_bounds(patch_path)
    lon = (left + right) / 2
    lat = (bottom + top) / 2
    ds = _dem_dataset(lon, lat, cfg)

    min_window = int(cfg.get("min_window_px", 11))
    half_deg_x = max((right - left) / 2, abs(ds.transform.a) * min_window / 2)
    half_deg_y = max((top - bottom) / 2, abs(ds.transform.e) * min_window / 2)
    window = from_bounds(lon - half_deg_x, lat - half_deg_y,
                          lon + half_deg_x, lat + half_deg_y,
                          transform=ds.transform)
    dem = ds.read(1, window=window, boundless=True, fill_value=np.nan)
    stats = slope_stats(dem, lat, ds.transform.a, ds.transform.e, nodata=ds.nodata)
    stats["dem_tile"] = dem_tile_name(lon, lat)
    return stats
