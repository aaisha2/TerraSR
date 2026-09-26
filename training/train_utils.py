"""Shared training utilities for stages 7-8."""
import csv
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda=True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def psnr(sr: torch.Tensor, hr: torch.Tensor, max_val: float = 1.0) -> float:
    """Mean PSNR (dB) over a batch of images in [0, max_val]."""
    mse = torch.mean((sr.clamp(0, max_val) - hr) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10)
    return torch.mean(10 * torch.log10((max_val ** 2) / mse)).item()


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count if self.count else 0.0


def capture_rng_state() -> dict:
    """Snapshot the RNG state so a resumed run continues the same random
    sequence (augmentation, shuffling) rather than restarting it."""
    state = {
        "torch": torch.get_rng_state(),
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict):
    if not state:
        return
    try:
        torch.set_rng_state(state["torch"])
        np.random.set_state(state["numpy"])
        random.setstate(state["python"])
        if state.get("cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["cuda"])
    except Exception as e:   # never let RNG restore break a resume
        print(f"warning: could not restore RNG state ({e}); continuing")


def save_checkpoint(path, model, optimizer, epoch, best_metric, extra=None):
    """Atomically write a full checkpoint (model + optimizer + epoch +
    best_metric + RNG). Atomic = write to a temp file then os.replace, so a
    crash or power loss mid-write can never corrupt an existing checkpoint -
    you either keep the previous good file or get the complete new one."""
    payload = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_metric": best_metric,
        "rng_state": capture_rng_state(),
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)   # atomic on the same filesystem (POSIX + Windows)


def load_checkpoint(path, model, optimizer=None, map_location="cpu"):
    # weights_only=False: these are our own training checkpoints and carry
    # optimizer + RNG state (incl. numpy), which the PyTorch 2.6+ safe loader
    # rejects. Safe here because we produced the file ourselves.
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


# --------------------------------------------------------------------------
# progress reporting
# --------------------------------------------------------------------------
EPOCH_LOG_NAME = "training_log.csv"
EPOCH_LOG_COLUMNS = ["timestamp", "epoch", "total_epochs", "train_loss", "val_psnr",
                     "best_psnr", "is_best", "epoch_seconds", "notes"]


def fmt_duration(seconds: float) -> str:
    """'2h13m' / '7m41s' / '52s' - short enough to sit at the end of a line."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


class ProgressReporter:
    """Progress output for a long training run.

    Two things silently swallow training progress, and both are handled here:
      - stdout buffering. The Colab notebook launches the trainer as a
        subprocess, so Python buffers stdout (it is not a TTY) and nothing
        appears until the process exits. Every line here is flushed.
      - lost cell output. Colab discards a cell's output when the browser tab
        disconnects, so the epoch lines can vanish even though training ran.
        Every epoch is therefore also appended to <out_dir>/training_log.csv,
        which lives on Drive and can be read later (or from another session)
        with training/training_status.py.
    """

    def __init__(self, out_dir, total_epochs: int, log_every: int = 50):
        self.out_dir = Path(out_dir)
        self.total_epochs = total_epochs
        self.log_every = int(log_every)
        self.log_path = self.out_dir / EPOCH_LOG_NAME
        self.run_start = time.time()
        self._epoch_seconds = []
        self._epoch = 0
        self._n_batches = 0
        self._interval = 0
        self._epoch_start = 0.0

    def info(self, msg: str):
        print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)

    def header(self, lines):
        """Run banner: what is being trained, from where, and how much is left."""
        print("=" * 62, flush=True)
        for line in lines:
            print(line, flush=True)
        print("=" * 62, flush=True)

    def start_epoch(self, epoch: int, n_batches: int):
        self._epoch = epoch
        self._n_batches = n_batches
        self._epoch_start = time.time()
        # short epochs should still show a couple of lines, long ones shouldn't flood
        self._interval = max(1, min(self.log_every, max(1, n_batches // 4)))
        self.info(f"epoch {epoch}/{self.total_epochs} started ({n_batches} batches)")

    def batch(self, batch_idx: int, loss: float):
        """Called after every optimizer step; prints on the reporting interval
        so a multi-minute epoch visibly moves instead of looking hung."""
        if batch_idx % self._interval and batch_idx != self._n_batches:
            return
        elapsed = time.time() - self._epoch_start
        rate = batch_idx / elapsed if elapsed > 0 else 0.0
        eta = (self._n_batches - batch_idx) / rate if rate > 0 else 0.0
        pct = 100 * batch_idx / self._n_batches if self._n_batches else 100
        self.info(f"  epoch {self._epoch}/{self.total_epochs}  "
                  f"batch {batch_idx}/{self._n_batches} ({pct:3.0f}%)  "
                  f"loss {loss:.4f}  {rate:.2f} it/s  epoch ETA {fmt_duration(eta)}")

    def end_epoch(self, epoch, train_loss, val_psnr=None, best_psnr=None,
                  is_best=False, notes=""):
        epoch_seconds = time.time() - self._epoch_start
        self._epoch_seconds.append(epoch_seconds)

        msg = (f"epoch {epoch:3d}/{self.total_epochs} done  "
               f"loss {train_loss:.4f}")
        if notes:
            msg += f"  [{notes}]"
        if val_psnr is not None:
            msg += f"  val PSNR {val_psnr:.2f} dB"
        if best_psnr is not None and best_psnr > -1:
            msg += f"  best {best_psnr:.2f} dB"
        if is_best:
            msg += "  <- best, best.pth updated"
        msg += f"  ({fmt_duration(epoch_seconds)})"

        remaining = self.total_epochs - epoch
        if remaining > 0:
            mean = sum(self._epoch_seconds) / len(self._epoch_seconds)
            msg += (f"  |  {remaining} epoch(s) left, "
                    f"ETA {fmt_duration(remaining * mean)}")
        self.info(msg)

        self._append_log({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "epoch": epoch,
            "total_epochs": self.total_epochs,
            "train_loss": f"{train_loss:.6f}",
            "val_psnr": "" if val_psnr is None else f"{val_psnr:.4f}",
            "best_psnr": "" if best_psnr is None else f"{best_psnr:.4f}",
            "is_best": int(bool(is_best)),
            "epoch_seconds": f"{epoch_seconds:.1f}",
            "notes": notes,
        })

    def finish(self, best_psnr):
        self.info(f"training complete: {self.total_epochs} epochs, "
                  f"best val PSNR {best_psnr:.2f} dB, "
                  f"total {fmt_duration(time.time() - self.run_start)}")
        self.info(f"checkpoints -> {self.out_dir}  |  epoch log -> {self.log_path}")

    def _append_log(self, row: dict):
        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            new = not self.log_path.exists() or self.log_path.stat().st_size == 0
            with open(self.log_path, "a", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=EPOCH_LOG_COLUMNS)
                if new:
                    w.writeheader()
                w.writerow(row)
        except Exception as e:    # a logging failure must never kill training
            print(f"warning: could not append to {self.log_path} ({e})", flush=True)


def checkpoint_progress(out_dir) -> dict:
    """Read how far a run got, without building the model: epoch reached, best
    val PSNR and where it came from. Used by training/training_status.py and by
    the trainers to report their own resume point."""
    out_dir = Path(out_dir)
    info = {"out_dir": str(out_dir), "exists": False}
    last, best = out_dir / "last.pth", out_dir / "best.pth"
    if not last.exists() and not best.exists():
        return info
    info["exists"] = True
    for key, path in (("last", last), ("best", best)):
        if not path.exists():
            continue
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            info[f"{key}_error"] = str(e)
            continue
        info[f"{key}_epoch"] = ckpt.get("epoch")
        info[f"{key}_metric"] = ckpt.get("best_metric")
        info["model_name"] = ckpt.get("model_name", info.get("model_name"))
        cfg = ckpt.get("config") or {}
        total = (cfg.get("train") or {}).get("epochs")
        if total:
            info["total_epochs"] = total
        info[f"{key}_mtime"] = datetime.fromtimestamp(
            path.stat().st_mtime).isoformat(timespec="seconds")
    log_path = out_dir / EPOCH_LOG_NAME
    if log_path.exists():
        info["log_path"] = str(log_path)
    return info
