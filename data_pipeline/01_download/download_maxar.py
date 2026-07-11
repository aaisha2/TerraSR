"""Stage 1: download Maxar Open Data PAN (pan_analytic) tiles.

Walks the public STAC catalog at maxar-opendata (verified 2026-07-11 structure):

    events/catalog.json
      -> events/<event>/collection.json
        -> events/<event>/ard/acquisition_collections/<catalog_id>_collection.json
          -> events/<event>/ard/<utm_zone>/<quadkey>/<date>/<catalog_id>.json  (STAC item)
            -> assets.pan_analytic.href = "./<catalog_id>-pan.tif"

No AWS account or credentials needed — plain public HTTPS/S3 GET.

Usage:
    python download_maxar.py --config configs/datasets.yaml --list-only
    python download_maxar.py --config configs/datasets.yaml --event Brazil-Flooding-May24 --max-files 1
"""
import argparse
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests
import yaml

sys.path.insert(0, str(Path(__file__).parent))
from _common import human_size, stream_download_url  # noqa: E402


def get_json(url: str) -> dict:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()


def iter_acquisition_collections(event_collection_url: str):
    coll = get_json(event_collection_url)
    for link in coll.get("links", []):
        if link.get("rel") == "child":
            yield urljoin(event_collection_url, link["href"])


def iter_items(acquisition_collection_url: str):
    coll = get_json(acquisition_collection_url)
    for link in coll.get("links", []):
        if link.get("rel") == "item":
            yield urljoin(acquisition_collection_url, link["href"])


def iter_pan_assets(event: str, catalog_url: str, asset_key: str):
    """Yield (item_url, asset_href, size_or_None) for every STAC item under
    the given event that has the requested asset."""
    catalog = get_json(catalog_url)
    event_href = None
    for link in catalog.get("links", []):
        if link.get("rel") == "child" and event in link["href"]:
            event_href = urljoin(catalog_url, link["href"])
            break
    if event_href is None:
        print(f"  event '{event}' not found in catalog — check spelling against catalog.json")
        return

    for acq_url in iter_acquisition_collections(event_href):
        for item_url in iter_items(acq_url):
            item = get_json(item_url)
            asset = item.get("assets", {}).get(asset_key)
            if asset is None:
                continue
            asset_url = urljoin(item_url, asset["href"])
            size = asset.get("file:size")  # not always present
            yield item_url, asset_url, size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=Path("configs/datasets.yaml"), type=Path)
    ap.add_argument("--event", action="append", default=None,
                     help="restrict to one or more events (repeatable); default: all in config")
    ap.add_argument("--max-files", type=int, default=None,
                     help="cap number of tiles downloaded per event (useful for a first smoke test)")
    ap.add_argument("--list-only", action="store_true",
                     help="list matching tiles without downloading anything")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["maxar"]
    catalog_url = cfg["catalog_url"]
    asset_key = cfg["asset"]
    out_dir = Path(cfg["out_dir"])
    events = args.event or cfg["events"]

    grand_total_files = 0
    for event in events:
        print(f"\n=== {event} ({asset_key}) ===")
        n = 0
        for item_url, asset_url, size in iter_pan_assets(event, catalog_url, asset_key):
            if args.max_files is not None and n >= args.max_files:
                break
            n += 1
            grand_total_files += 1
            fname = Path(asset_url).name
            dest = out_dir / event / fname
            if args.list_only:
                size_str = human_size(size) if size else "size unknown"
                print(f"  {asset_url}  ({size_str})")
            else:
                print(f"  {fname}")
                stream_download_url(asset_url, dest, expected_size=size)
        if n == 0:
            print(f"  no items with asset '{asset_key}' found for event '{event}'")

    verb = "would fetch" if args.list_only else "fetched"
    print(f"\n{verb} {grand_total_files} tiles")


if __name__ == "__main__":
    main()
