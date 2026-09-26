"""Stage 7: train a baseline SR model (SRCNN / SRGAN-SRResNet / SwinIR) on the
stage 6 dataset. Model-agnostic - the model is selected by config, everything
else (data, loss, loop) is shared, so benchmarking three baselines under
identical conditions is three config files (docs/plan §7).

Usage:
    python training/train_baseline.py --config configs/train_baseline.yaml
    python training/train_baseline.py --config configs/train_baseline.yaml \
        --override model.name=srcnn train.epochs=5
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import models  # noqa: E402
from terrasr_data import TerraSRDataset, load_terrain_index  # noqa: E402
from training.train_utils import (AverageMeter, ProgressReporter, get_device,  # noqa: E402
                                    load_checkpoint, psnr, restore_rng_state,
                                    save_checkpoint, set_seed)


def apply_overrides(cfg: dict, overrides: list):
    """Apply dotted key=value overrides, e.g. model.name=srcnn train.epochs=5."""
    for item in overrides or []:
        key, _, val = item.partition("=")
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        # best-effort scalar typing
        try:
            val = yaml.safe_load(val)
        except Exception:
            pass
        node[parts[-1]] = val
    return cfg


def make_loader(csv_path, terrain_index, split, cfg, train):
    ds = TerraSRDataset(csv_path, terrain_index=terrain_index, split=None,
                        normalize=cfg["data"]["normalize"],
                        augment=cfg["data"]["augment"] and train,
                        pan_filter=cfg["data"].get("pan_filter", "all"))
    return DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=train,
                      num_workers=cfg["train"]["num_workers"], drop_last=train)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    meter = AverageMeter()
    for lr, hr, _, _ in loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr)
        meter.update(psnr(sr, hr), n=lr.size(0))
    return meter.avg


def train(cfg, fresh=False):
    total_epochs = cfg["train"]["epochs"]
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = ProgressReporter(out_dir, total_epochs,
                           log_every=cfg["train"].get("log_every", 50))

    set_seed(cfg["train"]["seed"])
    device = get_device()

    terrain_index = load_terrain_index(cfg["data"]["terrain_config"])
    train_loader = make_loader(cfg["data"]["train_csv"], terrain_index, "train", cfg, train=True)
    val_loader = make_loader(cfg["data"]["val_csv"], terrain_index, "val", cfg, train=False)

    model = models.build(cfg["model"]["name"], cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = nn.L1Loss() if cfg["train"]["loss"] == "l1" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    extra = {"model_name": cfg["model"]["name"], "config": cfg}

    # resume from last.pth if present (unless --fresh). This is what makes a
    # crash / power loss recoverable: last.pth is saved every epoch below.
    start_epoch = 1
    best_psnr = -1.0
    last_path = out_dir / "last.pth"
    resumed = False
    if last_path.exists() and not fresh:
        ckpt = load_checkpoint(last_path, model, optimizer, map_location=device)
        start_epoch = ckpt["epoch"] + 1
        best_psnr = ckpt.get("best_metric", -1.0)
        restore_rng_state(ckpt.get("rng_state"))
        resumed = True

    banner = [
        f"model      : {cfg['model']['name']}  ({n_params/1e6:.2f}M params)",
        f"device     : {device}",
        f"data       : {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val patches, batch {cfg['train']['batch_size']}"
        f" -> {len(train_loader)} batches/epoch",
        f"out_dir    : {out_dir}",
    ]
    if resumed:
        done = start_epoch - 1
        banner += [
            f"RESUMING   : {done}/{total_epochs} epochs already done "
            f"({100*done/total_epochs:.0f}%) - continuing at epoch {start_epoch}",
            f"best so far: val PSNR {best_psnr:.2f} dB",
            f"to train   : {max(0, total_epochs - done)} more epoch(s)",
        ]
    else:
        banner += [f"starting fresh: epoch 1 -> {total_epochs}"
                   + ("  (--fresh: ignoring existing last.pth)"
                      if fresh and last_path.exists() else "")]
    rep.header(banner)

    if start_epoch > total_epochs:
        rep.info(f"already trained all {total_epochs} epochs (last.pth at epoch "
                 f"{start_epoch-1}, best val PSNR {best_psnr:.2f} dB); nothing to do. "
                 f"Use --fresh to retrain.")
        return

    n_batches = len(train_loader)
    for epoch in range(start_epoch, total_epochs + 1):
        rep.start_epoch(epoch, n_batches)
        model.train()
        loss_meter = AverageMeter()
        for i, (lr, hr, _, _) in enumerate(train_loader, start=1):
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), n=lr.size(0))
            rep.batch(i, loss_meter.avg)

        val_psnr, is_best = None, False
        if epoch % cfg["train"]["val_every"] == 0:
            val_psnr = validate(model, val_loader, device)
            if val_psnr > best_psnr:                       # best.pth: only on improvement
                best_psnr = val_psnr
                is_best = True
                save_checkpoint(out_dir / "best.pth", model, optimizer, epoch, best_psnr, extra=extra)
        # last.pth: EVERY epoch, so training can resume exactly where it stopped
        save_checkpoint(last_path, model, optimizer, epoch, best_psnr, extra=extra)
        rep.end_epoch(epoch, loss_meter.avg, val_psnr=val_psnr,
                      best_psnr=best_psnr, is_best=is_best)

    rep.finish(best_psnr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--override", nargs="*", default=[],
                     help="dotted overrides, e.g. model.name=srcnn train.epochs=5")
    ap.add_argument("--fresh", action="store_true",
                     help="ignore any existing last.pth and start training from scratch")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cfg = apply_overrides(cfg, args.override)
    train(cfg, fresh=args.fresh)


if __name__ == "__main__":
    main()
