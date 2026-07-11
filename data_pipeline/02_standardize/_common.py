"""Shared raster I/O helpers for stage 2 standardization scripts."""
import math

import numpy as np
import rasterio

STANDARD_DTYPE = "uint16"
STANDARD_NODATA = 0  # sentinel; real DN=0 is vanishingly rare in valid PAN scenes


def utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    """EPSG code for the UTM zone containing (lon, lat)."""
    zone = int(math.floor((lon + 180) / 6) + 1)
    return (32600 if lat >= 0 else 32700) + zone


def is_geographic_crs(crs) -> bool:
    return crs is not None and crs.is_geographic


def cast_to_standard_dtype(arr: np.ndarray, src_dtype: str) -> np.ndarray:
    """Normalize any input dtype to STANDARD_DTYPE (uint16) without discarding
    dynamic range: uint8 sources are rescaled up to fill the 16-bit range so a
    later shared degradation/normalization step can treat every source the
    same way; uint16 sources (true PAN, already sensor-native DN) pass through
    unchanged so radiometric calibration isn't silently altered."""
    if src_dtype == "uint16":
        return arr
    if src_dtype == "uint8":
        return (arr.astype(np.uint32) * 257).astype(np.uint16)  # 255*257 = 65535
    if np.issubdtype(np.dtype(src_dtype), np.floating):
        clipped = np.clip(arr, 0.0, 1.0)
        return (clipped * 65535).astype(np.uint16)
    raise ValueError(f"unhandled source dtype for standardization: {src_dtype}")


def write_standard_geotiff(path, data: np.ndarray, transform, crs, tags: dict) -> None:
    """Write a single-band, tiled, DEFLATE-compressed GeoTIFF with the
    standardized dtype/nodata and the given metadata tags."""
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": STANDARD_DTYPE,
        "crs": crs,
        "transform": transform,
        "nodata": STANDARD_NODATA,
        "compress": "deflate",
        "predictor": 2,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(**tags)
