"""Stage 9c: downstream-task evaluation — does SR enhancement actually help a
detector? (Proposal objective: "impact of enhanced imagery on selected remote
sensing tasks".)

The rigorous version needs ground-truth object boxes (e.g. SpaceNet building
footprints) and a detector fine-tuned on them, then reports mAP on LR vs SR vs
HR. That GT + fine-tuned detector isn't wired up yet, so this harness reports a
runnable PROXY: using a pretrained detector as a fixed reference, it measures
how well each SR output reproduces the detections the detector makes on the
true HR image (detection-consistency). The idea: if SR restores the structure
a detector relies on, its detections on SR should match those on HR far better
than its detections on the (upsampled) LR do.

This is explicitly a proxy — documented as such — until real GT boxes land.
Swap `reference_detector` for the SpaceNet-fine-tuned detector and replace the
consistency metric with mAP-vs-GT to get the final number.

Usage:
    python evaluation/eval_downstream_detection.py \
        --test-csv data/dataset/test.csv \
        --checkpoint checkpoints/terrasr/best.pth
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation._eval_common import (load_model_from_checkpoint,  # noqa: E402
                                       make_test_loader)
from training.train_utils import get_device  # noqa: E402


def load_reference_detector(device):
    """Pretrained COCO Faster R-CNN as a fixed reference detector. Placeholder
    for a SpaceNet-fine-tuned detector — see module docstring."""
    from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights,
                                               fasterrcnn_resnet50_fpn)
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=0.3)
    return model.eval().to(device)


def _to_rgb_uint(img_1chw):
    # detector expects 3-channel float [0,1]; PAN -> replicate
    return img_1chw.clamp(0, 1).repeat(3, 1, 1)


@torch.no_grad()
def detection_consistency(detector, ref_img, test_img, device, iou_thresh=0.5):
    """Fraction of the detector's HR (reference) boxes that are matched by a
    box in the test image (same crude IoU match). 1.0 = the test image
    reproduces every structure the detector found on HR."""
    ref = detector([_to_rgb_uint(ref_img).to(device)])[0]
    test = detector([_to_rgb_uint(test_img).to(device)])[0]
    ref_boxes = ref["boxes"].cpu()
    test_boxes = test["boxes"].cpu()
    if len(ref_boxes) == 0:
        return None  # no reference structure to reproduce; skip this patch
    if len(test_boxes) == 0:
        return 0.0

    from torchvision.ops import box_iou
    iou = box_iou(ref_boxes, test_boxes)
    matched = (iou.max(dim=1).values >= iou_thresh).float().mean().item()
    return matched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-csv", required=True, type=Path)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--terrain-config", default="configs/terrain_classes.yaml")
    ap.add_argument("--bicubic-scale", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="cap patches (debug)")
    args = ap.parse_args()

    device = get_device()
    print(f"device: {device}")
    print("NOTE: proxy metric (detection-consistency vs HR), not mAP-vs-GT — "
          "see module docstring.\n")

    loader = make_test_loader(args.test_csv, args.terrain_config, batch_size=1)
    detector = load_reference_detector(device)
    model, name, is_terrain = load_model_from_checkpoint(args.checkpoint, device)

    lr_scores, sr_scores = [], []
    for i, (lr, hr, terrain_idx, meta) in enumerate(loader):
        if args.limit and i >= args.limit:
            break
        lr, hr = lr.to(device), hr.to(device)
        sr = model(lr, terrain_idx.to(device)) if is_terrain else model(lr)
        bicubic = F.interpolate(lr, scale_factor=args.bicubic_scale,
                                mode="bicubic", align_corners=False)

        c_lr = detection_consistency(detector, hr[0], bicubic[0], device)
        c_sr = detection_consistency(detector, hr[0], sr[0], device)
        if c_lr is not None:
            lr_scores.append(c_lr)
        if c_sr is not None:
            sr_scores.append(c_sr)

    def mean(xs):
        return float(np.mean(xs)) if xs else float("nan")

    print(f"patches with reference detections: {len(sr_scores)}")
    print(f"detection-consistency vs HR:")
    print(f"  bicubic(LR): {mean(lr_scores):.3f}")
    print(f"  {name} SR:   {mean(sr_scores):.3f}")
    print("\n(higher = SR better reproduces the structure the detector uses on HR)")


if __name__ == "__main__":
    main()
