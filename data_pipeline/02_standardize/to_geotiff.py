"""Stage 2b: standardize a single-band raster (already run through
extract_pan_band.py) into the common format every later stage assumes:

  - GeoTIFF, single band
  - projected CRS in meters — reprojects geographic (lon/lat degree) sources
    to their local UTM zone; leaves already-projected sources untouched
    (no resampling loss for imagery that's already in a metric CRS, which
    is the common case for SpaceNet/Maxar)
  - dtype normalized to uint16 (see _common.cast_to_standard_dtype for the
    exact rule per source dtype)
  - tiled + DEFLATE-compressed, nodata=0

This is the script that resolves the "standardize a dataset from multiple
sources" question raised with the supervisor on 2026-07-09.

Usage:
    python to_geotiff.py --in extracted.tif --out standardized.tif
"""
import argparse
from pathlib import Path

import rasterio
from rasterio.warp import Resampling, calculate_default_transform, reproject

from _common import (cast_to_standard_dtype, is_geographic_crs,
                      utm_epsg_for_lonlat, write_standard_geotiff)


def reproject_to_utm(src: rasterio.DatasetReader):
    """Reproject a geographic-CRS raster to its local UTM zone, computed
    from the scene centroid. Returns (data, transform, dst_crs)."""
    lon = (src.bounds.left + src.bounds.right) / 2
    lat = (src.bounds.bottom + src.bounds.top) / 2
    dst_crs = f"EPSG:{utm_epsg_for_lonlat(lon, lat)}"

    dst_transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)

    import numpy as np
    dst_data = np.zeros((height, width), dtype=src.dtypes[0])
    reproject(
        source=rasterio.band(src, 1),
        destination=dst_data,
        src_transform=src.transform,
        src_crs=src.crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )
    return dst_data, dst_transform, dst_crs


def standardize_to_file(in_path: Path, out_path: Path) -> dict:
    """Standardize a single-band raster (CRS + dtype) and write it to out_path.
    Reusable by the batch driver (standardize_scenes.py) and the CLI below."""
    with rasterio.open(in_path) as src:
        if src.count != 1:
            raise ValueError(f"expected a single-band input (run extract_pan_band.py "
                              f"first), got {src.count} bands")
        src_dtype = str(src.dtypes[0])
        prior_tags = src.tags()
        orig_crs = str(src.crs)

        if is_geographic_crs(src.crs):
            data, transform, crs = reproject_to_utm(src)
            reprojected = True
        else:
            data, transform, crs = src.read(1), src.transform, src.crs
            reprojected = False

    standardized = cast_to_standard_dtype(data, src_dtype)

    tags = dict(prior_tags)
    tags.update({
        "standardized": "true",
        "orig_dtype": src_dtype,
        "reprojected_to_utm": str(reprojected).lower(),
        "orig_crs": orig_crs,
    })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_standard_geotiff(out_path, standardized, transform, crs, tags)

    return {"crs": str(crs), "dtype": str(standardized.dtype),
            "reprojected": reprojected, "shape": standardized.shape}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, type=Path)
    ap.add_argument("--out", dest="out_path", required=True, type=Path)
    args = ap.parse_args()

    info = standardize_to_file(args.in_path, args.out_path)
    print(f"{args.in_path.name} -> {args.out_path.name}  crs={info['crs']}  "
          f"dtype={info['dtype']}  reprojected={info['reprojected']}  shape={info['shape']}")


if __name__ == "__main__":
    main()
