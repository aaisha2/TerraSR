"""Shared helpers for stage 1 download scripts. Not a stage of its own —
imported by download_spacenet.py / download_maxar.py.
"""
import sys
from pathlib import Path

import requests


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def already_downloaded(dest: Path, expected_size: int | None) -> bool:
    """Idempotency check: skip files already on disk with a matching size.
    A mismatched size means a previous download was interrupted — re-fetch."""
    if not dest.exists():
        return False
    if expected_size is None:
        return True
    return dest.stat().st_size == expected_size


def stream_download_url(url: str, dest: Path, expected_size: int | None = None,
                         chunk_size: int = 1 << 20, timeout: int = 60) -> None:
    """Download a URL to `dest` with streaming writes, skipping if already
    present with the right size. Writes to a .part file first so a killed
    download never looks "complete" to `already_downloaded`."""
    if already_downloaded(dest, expected_size):
        print(f"  skip (already downloaded): {dest.name}")
        return

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", expected_size or 0))
        written = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                written += len(chunk)
                if total:
                    pct = 100 * written / total
                    print(f"\r  {dest.name}: {human_size(written)}/{human_size(total)} ({pct:.0f}%)",
                          end="", flush=True)
    print()
    tmp.rename(dest)
