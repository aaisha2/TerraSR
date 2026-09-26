"""How far has training got? Reads the checkpoints written by
train_baseline.py / train_terrasr.py and reports, per model, the epoch reached,
the best validation PSNR, and the last few epoch lines.

Nothing here touches the GPU or builds a model, so it is safe to run while
training is in progress (or in a second Colab notebook against the same Drive
folder) - useful because Colab throws away a cell's output when the browser tab
disconnects, while checkpoints and training_log.csv survive.

Usage:
    python training/training_status.py                       # all of checkpoints/
    python training/training_status.py --dir checkpoints/terrasr --tail 20
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.train_utils import EPOCH_LOG_NAME, checkpoint_progress  # noqa: E402


def find_run_dirs(root: Path):
    """A run directory is one holding last.pth / best.pth - either `root`
    itself or its immediate children (checkpoints/srcnn, checkpoints/terrasr...)."""
    if (root / "last.pth").exists() or (root / "best.pth").exists():
        return [root]
    return sorted(d for d in root.iterdir()
                  if d.is_dir() and ((d / "last.pth").exists() or (d / "best.pth").exists()))


def tail_log(run_dir: Path, n: int):
    log_path = run_dir / EPOCH_LOG_NAME
    if not log_path.exists():
        return []
    with open(log_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-n:] if n else []


def report(run_dir: Path, tail: int):
    info = checkpoint_progress(run_dir)
    name = info.get("model_name") or run_dir.name
    print(f"\n{run_dir.name}  (model: {name})")
    print("-" * 62)
    if not info["exists"]:
        print("  no checkpoint yet - training has not written an epoch")
        return

    total = info.get("total_epochs")
    last_epoch = info.get("last_epoch")
    if last_epoch is not None:
        pct = f" ({100*last_epoch/total:.0f}%)" if total else ""
        done = f"{last_epoch}/{total}" if total else str(last_epoch)
        print(f"  epochs done   : {done}{pct}")
        print(f"  last.pth saved: {info.get('last_mtime')}")
        if total and last_epoch >= total:
            print("  status        : finished - re-running the training cell will skip it")
        else:
            nxt = last_epoch + 1
            print(f"  status        : will resume at epoch {nxt}"
                  + (f" ({total - last_epoch} left)" if total else ""))
    if info.get("best_epoch") is not None:
        print(f"  best val PSNR : {info['best_metric']:.2f} dB "
              f"(epoch {info['best_epoch']}, saved {info.get('best_mtime')})")
    for key in ("last_error", "best_error"):
        if info.get(key):
            print(f"  warning       : could not read {key.split('_')[0]}.pth: {info[key]}")

    rows = tail_log(run_dir, tail)
    if rows:
        print(f"  last {len(rows)} epoch(s) from {EPOCH_LOG_NAME}:")
        for r in rows:
            line = (f"    epoch {r['epoch']:>4}/{r['total_epochs']}  "
                    f"loss {r['train_loss']}  {r['epoch_seconds']}s")
            if r.get("val_psnr"):
                line += f"  val PSNR {float(r['val_psnr']):.2f} dB"
            if r.get("is_best") == "1":
                line += "  <- best"
            print(f"{line}   {r['timestamp']}")
    elif (run_dir / EPOCH_LOG_NAME).exists():
        print(f"  {EPOCH_LOG_NAME} is empty")
    else:
        print(f"  no {EPOCH_LOG_NAME} yet (it is written from the first completed "
              f"epoch of a run started with this version of the trainer)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("checkpoints"),
                    help="a run directory, or a parent holding one per model")
    ap.add_argument("--tail", type=int, default=5,
                    help="how many recent epoch log lines to show (0 = none)")
    args = ap.parse_args()

    if not args.dir.exists():
        print(f"{args.dir} does not exist - no training has been started yet.")
        return
    run_dirs = find_run_dirs(args.dir)
    if not run_dirs:
        print(f"no checkpoints found under {args.dir} - no epoch has completed yet.")
        return
    print(f"training status under {args.dir.resolve()}")
    for d in run_dirs:
        report(d, args.tail)
    print()


if __name__ == "__main__":
    main()
