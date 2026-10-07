"""Add patch geography (center_lon, center_lat, bounds_wgs84) to a manifest
that was built before stage 3 recorded it.

The stage 6b split needs to know where each patch is on the ground, because
splitting by scene name does not prevent same-location leakage. New patchify
runs record it; this backfills an existing dataset by reading the geotransform
off each patch GeoTIFF, so a 20k-patch dataset does not have to be rebuilt.

Works on a stage 3/4 JSON manifest or a stage 6 CSV/Parquet manifest.

Usage:
    python backfill_patch_geo.py --manifest data/patches/patch_manifest_labeled.json
    python backfill_patch_geo.py --manifest data/dataset/dataset_manifest.csv
"""
import argparse
import json
import os
from pathlib import Path

import rasterio
from rasterio.warp import transform_bounds

GEO_FIELDS = ("center_lon", "center_lat", "bounds_wgs84")
PROGRESS_EVERY = 2000


def geo_for_patch(path: str) -> dict:
    with rasterio.open(path) as ds:
        if ds.crs is None:
            return {"center_lon": None, "center_lat": None, "bounds_wgs84": None}
        left, bottom, right, top = transform_bounds(ds.crs, "EPSG:4326", *ds.bounds)
    return {"center_lon": (left + right) / 2, "center_lat": (bottom + top) / 2,
            "bounds_wgs84": [left, bottom, right, top]}


def path_for_row(row, prefer) -> str | None:
    for key in prefer:
        val = row.get(key) if isinstance(row, dict) else row[key]
        if val and str(val) != "nan" and Path(str(val)).exists():
            return str(val)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=None,
                     help="default: overwrite the input manifest in place")
    ap.add_argument("--force", action="store_true",
                     help="recompute geography even where it is already present")
    args = ap.parse_args()

    out = args.out or args.manifest
    # the patch itself is the most authoritative source; hr_path is the same
    # ground for a stage 6 manifest, which has no patch_path column
    prefer = ("patch_path", "hr_path")

    if args.manifest.suffix.lower() == ".json":
        rows = json.loads(args.manifest.read_text())
        n_done = n_skip = n_fail = 0
        for i, row in enumerate(rows, 1):
            if not args.force and row.get("center_lon") is not None:
                n_skip += 1
                continue
            path = path_for_row(row, prefer)
            if path is None:
                n_fail += 1
                continue
            try:
                row.update(geo_for_patch(path))
                n_done += 1
            except Exception as e:
                print(f"  FAILED {row.get('tile_id')}: {e}")
                n_fail += 1
            if i % PROGRESS_EVERY == 0:
                print(f"  {i}/{len(rows)} rows", flush=True)
        tmp = out.with_name(out.name + ".partial")
        tmp.write_text(json.dumps(rows, indent=2))
        os.replace(tmp, out)
        n_total = len(rows)
    else:
        import pandas as pd
        df = (pd.read_parquet(args.manifest) if args.manifest.suffix == ".parquet"
              else pd.read_csv(args.manifest))
        for f in GEO_FIELDS:
            if f not in df.columns:
                df[f] = None
        n_done = n_skip = n_fail = 0
        cache = {}
        for i, idx in enumerate(df.index, 1):
            if not args.force and df.at[idx, "center_lon"] == df.at[idx, "center_lon"] \
                    and df.at[idx, "center_lon"] is not None:
                n_skip += 1
                continue
            path = path_for_row(df.loc[idx], prefer)
            if path is None:
                n_fail += 1
                continue
            try:
                if path not in cache:
                    cache[path] = geo_for_patch(path)
                g = cache[path]
                df.at[idx, "center_lon"] = g["center_lon"]
                df.at[idx, "center_lat"] = g["center_lat"]
                df.at[idx, "bounds_wgs84"] = json.dumps(g["bounds_wgs84"])
                n_done += 1
            except Exception as e:
                print(f"  FAILED row {idx}: {e}")
                n_fail += 1
            if i % PROGRESS_EVERY == 0:
                print(f"  {i}/{len(df)} rows", flush=True)
        if out.suffix == ".parquet":
            df.to_parquet(out, index=False)
        else:
            df.to_csv(out, index=False)
        n_total = len(df)

    print(f"backfilled geography for {n_done} patches "
          f"({n_skip} already had it, {n_fail} could not be read) of {n_total}")
    print(f"-> {out}")
    if n_fail:
        print("  Patches whose file is missing keep empty geography; the stage 6b "
              "split and the leakage audit will report them rather than guess.")


if __name__ == "__main__":
    main()
