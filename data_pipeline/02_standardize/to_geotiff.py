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
from rasterio.vrt import WarpedVRT
from rasterio.warp import Resampling, calculate_default_transform

from _common import (STANDARD_NODATA, cast_to_standard_dtype, finalize_atomic,
                      is_geographic_crs, row_windows, standard_profile,
                      tmp_path_for, utm_epsg_for_lonlat)


def utm_grid(src: rasterio.DatasetReader):
    """Output grid for reprojecting a geographic-CRS raster to its local UTM
    zone (zone picked from the scene centroid). Returns (dst_crs, transform,
    width, height)."""
    lon = (src.bounds.left + src.bounds.right) / 2
    lat = (src.bounds.bottom + src.bounds.top) / 2
    dst_crs = f"EPSG:{utm_epsg_for_lonlat(lon, lat)}"
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)
    return dst_crs, transform, width, height


def standardize_to_file(in_path: Path, out_path: Path) -> dict:
    """Standardize a single-band raster (CRS + dtype) and write it to out_path.
    Reusable by the batch driver (standardize_scenes.py) and the CLI below.

    Streams the scene in row strips (reprojecting through a WarpedVRT when the
    source is geographic), so memory use is bounded no matter how large the
    scene is. The output appears only once fully written (atomic rename)."""
    with rasterio.open(in_path) as src:
        if src.count != 1:
            raise ValueError(f"expected a single-band input (run extract_pan_band.py "
                              f"first), got {src.count} bands")
        src_dtype = str(src.dtypes[0])
        prior_tags = src.tags()
        orig_crs = str(src.crs)
        reprojected = is_geographic_crs(src.crs)

        if reprojected:
            dst_crs, transform, width, height = utm_grid(src)
            reader = WarpedVRT(src, crs=dst_crs, transform=transform,
                               width=width, height=height,
                               resampling=Resampling.bilinear,
                               nodata=STANDARD_NODATA)
        else:
            reader = src
            dst_crs, transform, width, height = src.crs, src.transform, src.width, src.height

        tags = dict(prior_tags)
        tags.update({
            "standardized": "true",
            "orig_dtype": src_dtype,
            "reprojected_to_utm": str(reprojected).lower(),
            "orig_crs": orig_crs,
        })

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tmp_path_for(out_path)
        try:
            with rasterio.open(tmp, "w", **standard_profile(width, height, transform, dst_crs)) as dst:
                for win in row_windows(width, height):
                    dst.write(cast_to_standard_dtype(reader.read(1, window=win), src_dtype),
                              1, window=win)
                dst.update_tags(**tags)
        finally:
            if reader is not src:
                reader.close()
        finalize_atomic(tmp, out_path)

    return {"crs": str(dst_crs), "dtype": "uint16",
            "reprojected": reprojected, "shape": (height, width)}


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
