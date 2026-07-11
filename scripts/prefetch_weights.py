"""Pre-download the pretrained weights the training/eval stages pull from the
torch hub, so a full RESOLVE run doesn't stall mid-stage (and so a node with
restricted outbound access fetches them once, up front).

  - VGG16 (ImageNet)          — perceptual term of the terrain-aware loss (stage 8)
  - Faster R-CNN R50-FPN (COCO) — downstream-detection harness (stage 9c)

Both are cached under ~/.cache/torch and reused automatically afterwards.

    python scripts/prefetch_weights.py
"""
import sys


def prefetch_vgg16():
    from torchvision.models import VGG16_Weights, vgg16
    print("fetching VGG16 (ImageNet) for the perceptual loss ...")
    vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    print("  ok")


def prefetch_fasterrcnn():
    from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights,
                                               fasterrcnn_resnet50_fpn)
    print("fetching Faster R-CNN R50-FPN (COCO) for the detection harness ...")
    fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    print("  ok")


def main():
    prefetch_vgg16()
    prefetch_fasterrcnn()
    print("\nall pretrained weights cached.")


if __name__ == "__main__":
    sys.exit(main())
