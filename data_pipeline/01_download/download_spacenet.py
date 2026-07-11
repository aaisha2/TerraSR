"""Stage 1: download SpaceNet PAN (true panchromatic) imagery.

s3://spacenet-dataset is public and, as verified 2026-07-11, answers
anonymous/unsigned S3 requests directly — no AWS account, credentials, or
--request-payer flag needed (older docs describe it as requester-pays;
that is no longer enforced on the current bucket layout).

Bucket layout (verified):
    AOIs/<AOI_name>/PAN/<scene>.TIF

Usage:
    python download_spacenet.py --config configs/datasets.yaml --list-only
    python download_spacenet.py --config configs/datasets.yaml --aoi AOI_2_Vegas --max-files 2
"""
import argparse
import sys
from pathlib import Path

import boto3
import yaml
from botocore import UNSIGNED
from botocore.config import Config

sys.path.insert(0, str(Path(__file__).parent))
from _common import already_downloaded, human_size  # noqa: E402


def list_pan_scenes(s3, bucket: str, aoi: str, band: str):
    prefix = f"AOIs/{aoi}/{band}/"
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].upper().endswith(".TIF"):
                yield obj["Key"], obj["Size"]


def download_scene(s3, bucket: str, key: str, size: int, dest: Path) -> None:
    if already_downloaded(dest, size):
        print(f"  skip (already downloaded): {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    def progress(bytes_transferred, _total=[0]):
        _total[0] += bytes_transferred
        pct = 100 * _total[0] / size if size else 0
        print(f"\r  {dest.name}: {human_size(_total[0])}/{human_size(size)} ({pct:.0f}%)",
              end="", flush=True)

    s3.download_file(bucket, key, str(tmp), Callback=progress)
    print()
    tmp.rename(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=Path("configs/datasets.yaml"), type=Path)
    ap.add_argument("--aoi", action="append", default=None,
                     help="restrict to one or more AOIs (repeatable); default: all in config")
    ap.add_argument("--max-files", type=int, default=None,
                     help="cap number of scenes downloaded per AOI (useful for a first smoke test)")
    ap.add_argument("--list-only", action="store_true",
                     help="list matching scenes and sizes without downloading anything")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())["spacenet"]
    bucket = cfg["bucket"]
    band = cfg["band"]
    out_dir = Path(cfg["out_dir"])
    aois = args.aoi or cfg["aois"]

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    grand_total_bytes = 0
    grand_total_files = 0
    for aoi in aois:
        print(f"\n=== {aoi} ({band}) ===")
        n = 0
        for key, size in list_pan_scenes(s3, bucket, aoi, band):
            if args.max_files is not None and n >= args.max_files:
                break
            n += 1
            grand_total_bytes += size
            grand_total_files += 1
            dest = out_dir / aoi / Path(key).name
            if args.list_only:
                print(f"  {key}  ({human_size(size)})")
            else:
                download_scene(s3, bucket, key, size, dest)
        if n == 0:
            print(f"  no PAN scenes found under AOIs/{aoi}/{band}/ — check AOI name")

    verb = "would fetch" if args.list_only else "fetched"
    print(f"\n{verb} {grand_total_files} scenes, {human_size(grand_total_bytes)} total")


if __name__ == "__main__":
    main()
