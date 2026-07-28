"""Shared training utilities for stages 7-8."""
import os
import random
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
    crash or power loss mid-write can never corrupt an existing checkpoint —
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
