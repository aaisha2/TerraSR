"""Shared helpers for stage 3 patchify scripts."""
import numpy as np


def grid_windows(height: int, width: int, patch_size: int, stride: int, drop_partial_edge: bool):
    """Yield (row, col, row_off, col_off) for a fixed-size patch grid over an
    image of the given height/width. `row`/`col` are 0-indexed grid
    coordinates (used to build tile_id); `row_off`/`col_off` are pixel
    offsets for the rasterio Window."""
    row = 0
    row_off = 0
    while row_off < height:
        if row_off + patch_size > height:
            if drop_partial_edge:
                break
            row_off = max(0, height - patch_size)

        col = 0
        col_off = 0
        while col_off < width:
            if col_off + patch_size > width:
                if drop_partial_edge:
                    break
                col_off = max(0, width - patch_size)

            yield row, col, row_off, col_off

            if col_off + patch_size >= width:
                break
            col += 1
            col_off += stride

        if row_off + patch_size >= height:
            break
        row += 1
        row_off += stride


def patch_stats(arr: np.ndarray, nodata_value: int, saturated_percentile: float,
                 scene_max: float) -> dict:
    nodata_fraction = float(np.mean(arr == nodata_value))
    std_dev = float(np.std(arr))
    saturated_threshold = scene_max * (saturated_percentile / 100.0)
    saturated_fraction = float(np.mean(arr >= saturated_threshold)) if scene_max > 0 else 0.0
    return {
        "nodata_fraction": nodata_fraction,
        "std_dev": std_dev,
        "saturated_fraction": saturated_fraction,
    }
