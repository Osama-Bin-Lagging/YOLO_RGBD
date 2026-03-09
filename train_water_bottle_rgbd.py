#!/usr/bin/env python3
"""Unified RGBD training script with COCO pretrained weight transfer.

Supports earlyfusion, midfusion, and midfusion-P3 fusion strategies.
Handles the full pipeline: template → weight transfer → pretrained training.

Usage:
    python train_water_bottle_rgbd.py --fusion earlyfusion --device 0 --epochs 100
    python train_water_bottle_rgbd.py --fusion midfusion --device mps --epochs 100
    python train_water_bottle_rgbd.py --fusion midfusion-P3 --device 0
"""

import argparse
import warnings

import torch

warnings.filterwarnings("ignore")
from ultralytics import YOLO
from transform_COCO_to_RGBD import copy_and_modify_layers

# Layer mappings derived from YAML analysis of each fusion architecture.
# Standard yolo11n has layers 0-23 (backbone 0-10, head 11-23).
# Each fusion model prepends Silence + SilenceChannel layers and may duplicate
# the backbone for a second modality branch.
FUSION_CONFIGS = {
    "earlyfusion": {
        "yaml": "ultralytics/cfg/models/11-RGBD/yolo11-RGBD-earlyfusion.yaml",
        "use_simotm": "RGBD",
        "channels": 4,
        "copy_ranges": [
            ((0, 10), (2, 12)),    # backbone (+2 offset for Silence+SilenceChannel)
            ((11, 23), (13, 25)),  # head (+2 offset)
        ],
    },
    "midfusion": {
        "yaml": "ultralytics/cfg/models/11-RGBD/yolo11-RGBD-midfusion.yaml",
        "use_simotm": "RGBD",
        "channels": 4,
        "copy_ranges": [
            ((0, 8), (2, 10)),     # RGB backbone (layers 0-8 -> 2-10)
            ((0, 8), (12, 20)),    # IR backbone (layers 0-8 -> 12-20, first conv 3->1)
            ((9, 23), (24, 38)),   # SPPF + C2PSA + head
        ],
    },
    "midfusion-P3": {
        "yaml": "ultralytics/cfg/models/11-RGBD/yolo11-RGBD-midfusion-P3.yaml",
        "use_simotm": "RGBD",
        "channels": 4,
        "copy_ranges": [
            ((0, 4), (2, 6)),      # RGB branch up to P3
            ((0, 4), (8, 12)),     # IR branch up to P3 (first conv 3->1)
            ((5, 10), (15, 20)),   # shared P4+ backbone
            ((11, 23), (21, 33)),  # head
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(description="RGBD training with COCO pretrained weights")
    parser.add_argument("--fusion", choices=list(FUSION_CONFIGS.keys()), required=True,
                        help="Fusion strategy to use")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="0", help="Device: '0' for GPU, 'mps' for Mac")
    parser.add_argument("--pretrained", default="yolo11n.pt", help="COCO pretrained weights")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/water_bottle-rgbd.yaml",
                        help="Dataset YAML config")
    parser.add_argument("--project", default="runs/water_bottle", help="Project directory")
    parser.add_argument("--skip-template", action="store_true",
                        help="Skip template training if pretrained .pt already exists")
    args = parser.parse_args()

    cfg = FUSION_CONFIGS[args.fusion]
    use_amp = args.device != "mps"
    template_name = f"template-{args.fusion}"
    template_pt = f"{args.project}/{template_name}/weights/last.pt"
    pretrained_pt = f"yolo11n-RGBD-{args.fusion}-pretrained.pt"

    # --- Step 1: Train 1-epoch template to get model structure ---
    if not args.skip_template:
        print(f"\n{'='*60}")
        print(f"Step 1: Training 1-epoch template for {args.fusion}")
        print(f"{'='*60}\n")
        model = YOLO(cfg["yaml"])
        model.train(
            data=args.data,
            epochs=1,
            fraction=0.01,
            batch=args.batch,
            device=args.device,
            amp=use_amp,
            use_simotm=cfg["use_simotm"],
            channels=cfg["channels"],
            project=args.project,
            name=template_name,
        )
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # --- Step 2: Transfer COCO pretrained weights ---
    print(f"\n{'='*60}")
    print(f"Step 2: Transferring COCO weights to {args.fusion} model")
    print(f"{'='*60}\n")
    copy_and_modify_layers(
        source_model_path=args.pretrained,
        target_model_path=template_pt,
        output_model_path=pretrained_pt,
        copy_ranges=cfg["copy_ranges"],
    )

    # --- Step 3: Full training with pretrained weights ---
    print(f"\n{'='*60}")
    print(f"Step 3: Training {args.fusion} with pretrained weights ({args.epochs} epochs)")
    print(f"{'='*60}\n")
    model = YOLO(pretrained_pt)
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        optimizer="SGD",
        amp=use_amp,
        use_simotm=cfg["use_simotm"],
        channels=cfg["channels"],
        close_mosaic=10,
        workers=2,
        cache=False,
        imgsz=640,
        project=args.project,
        name=f"wb-yolo11n-RGBD-{args.fusion}-pretrained",
    )
    print(f"\nDone! Results saved to {args.project}/wb-yolo11n-RGBD-{args.fusion}-pretrained/")


if __name__ == "__main__":
    main()
