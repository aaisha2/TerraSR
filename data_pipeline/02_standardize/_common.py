"""Shared raster I/O helpers for stage 2 standardization scripts."""
import math
import os
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

STANDARD_DTYPE = "uint16"
STANDARD_NODATA = 0  # sentinel; real DN=0 is vanishingly rare in valid PAN scenes

# Rows processed per chunk. Scenes are streamed strip by strip so peak memory
# stays at a few hundred MB regardless of scene size -- a whole-scene read of a
# SpaceNet mosaic (57374 x 52845 uint16 = 6.1 GB) would exhaust a free Colab
# runtime's RAM, which Colab reports as a disconnect.
CHUNK_ROWS = 1024


def row_windows(width: int, height: int, chunk_rows: int = CHUNK_ROWS):
    """Yield full-width row-strip windows covering the raster."""
    for row_off in range(0, height, chunk_rows):
        yield Window(0, row_off, width, min(chunk_rows, height - row_off))


def tmp_path_for(path) -> Path:
    """Sibling temp path used for atomic writes. Kept as a .tif so GDAL still
    recognises the format; the final os.replace makes the output appear only
    once it is complete, so an interrupted run never leaves a truncated file
    that a later resume would mistake for finished work."""
    path = Path(path)
    return path.with_name(path.stem + ".partial" + path.suffix)


def finalize_atomic(tmp: Path, final) -> None:
    os.replace(tmp, final)


def standard_profile(width: int, height: int, transform, crs) -> dict:
    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
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
        "BIGTIFF": "IF_SAFER",
    }


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
    standardized dtype/nodata and the given metadata tags (atomically)."""
    profile = standard_profile(data.shape[1], data.shape[0], transform, crs)
    tmp = tmp_path_for(path)
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(data, 1)
        dst.update_tags(**tags)
    finalize_atomic(tmp, path)
