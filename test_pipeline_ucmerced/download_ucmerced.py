"""UC Merced test harness — stage 1 equivalent: download + extract the UC
Merced Land Use dataset (2100 images, 21 classes, 256x256, ~1ft RGB).

Uses the torchgeo HuggingFace mirror (the official weegee.vision.ucmerced.edu
link is frequently unreachable). Idempotent: skips download/extract if the
images are already present.

    python test_pipeline_ucmerced/download_ucmerced.py --out-dir test_pipeline_ucmerced/data/raw
"""
import argparse
import zipfile
from pathlib import Path

import requests

MIRRORS = [
    "https://huggingface.co/datasets/torchgeo/ucmerced/resolve/main/UCMerced_LandUse.zip",
    "http://weegee.vision.ucmerced.edu/datasets/UCMerced_LandUse.zip",
]


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".zip.part")
    with requests.get(url, stream=True, timeout=120,
                       headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                written += len(chunk)
                if total:
                    print(f"\r  {written/1e6:.0f}/{total/1e6:.0f} MB "
                          f"({100*written/total:.0f}%)", end="", flush=True)
                else:
                    print(f"\r  {written/1e6:.0f} MB", end="", flush=True)
    print()
    tmp.rename(dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    images_root = args.out_dir / "UCMerced_LandUse" / "Images"
    if images_root.exists() and any(images_root.iterdir()):
        n = sum(1 for _ in images_root.rglob("*.tif"))
        print(f"already extracted: {n} images under {images_root}")
        return

    zip_path = args.out_dir / "UCMerced_LandUse.zip"
    if not zip_path.exists():
        last_err = None
        for url in MIRRORS:
            try:
                print(f"downloading {url}")
                download(url, zip_path)
                break
            except Exception as e:
                last_err = e
                print(f"  mirror failed: {e}")
        else:
            raise SystemExit(f"all mirrors failed; last error: {last_err}")

    print(f"extracting {zip_path.name} ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(args.out_dir)

    n = sum(1 for _ in images_root.rglob("*.tif"))
    print(f"done: {n} images under {images_root}")


if __name__ == "__main__":
    main()
