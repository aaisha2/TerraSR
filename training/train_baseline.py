"""Stage 7: train a baseline SR model (SRCNN / SRGAN-SRResNet / SwinIR) on the
stage 6 dataset. Model-agnostic — the model is selected by config, everything
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
from training.train_utils import (AverageMeter, get_device, load_checkpoint,  # noqa: E402
                                    psnr, save_checkpoint, set_seed)


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
                        augment=cfg["data"]["augment"] and train)
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


def train(cfg):
    set_seed(cfg["train"]["seed"])
    device = get_device()
    print(f"device: {device}")

    terrain_index = load_terrain_index(cfg["data"]["terrain_config"])
    train_loader = make_loader(cfg["data"]["train_csv"], terrain_index, "train", cfg, train=True)
    val_loader = make_loader(cfg["data"]["val_csv"], terrain_index, "val", cfg, train=False)
    print(f"train patches: {len(train_loader.dataset)} | val patches: {len(val_loader.dataset)}")

    model = models.build(cfg["model"]["name"], cfg["model"]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {cfg['model']['name']}  ({n_params/1e6:.2f}M params)")

    criterion = nn.L1Loss() if cfg["train"]["loss"] == "l1" else nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])

    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    best_psnr = -1.0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        loss_meter = AverageMeter()
        for lr, hr, _, _ in train_loader:
            lr, hr = lr.to(device), hr.to(device)
            optimizer.zero_grad()
            sr = model(lr)
            loss = criterion(sr, hr)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), n=lr.size(0))

        msg = f"epoch {epoch:3d}/{cfg['train']['epochs']}  loss {loss_meter.avg:.4f}"
        if epoch % cfg["train"]["val_every"] == 0:
            val_psnr = validate(model, val_loader, device)
            msg += f"  val PSNR {val_psnr:.2f} dB"
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                save_checkpoint(out_dir / "best.pth", model, optimizer, epoch, best_psnr,
                                extra={"model_name": cfg["model"]["name"], "config": cfg})
                msg += "  <- best"
        print(msg)

    save_checkpoint(out_dir / "last.pth", model, optimizer, cfg["train"]["epochs"], best_psnr,
                    extra={"model_name": cfg["model"]["name"], "config": cfg})
    print(f"done. best val PSNR {best_psnr:.2f} dB. checkpoints -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--override", nargs="*", default=[],
                     help="dotted overrides, e.g. model.name=srcnn train.epochs=5")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    cfg = apply_overrides(cfg, args.override)
    train(cfg)


if __name__ == "__main__":
    main()
