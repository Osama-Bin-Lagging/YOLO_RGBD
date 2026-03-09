#!/usr/bin/env python3
"""Convert water_bottle RGBD dataset to YOLOv11-RGBD format.

Handles all water_bottle instances (water_bottle_1 through water_bottle_10).
Scans rgbd-dataset/water_bottle/ for all instance directories.
"""

import os
import random
import re
import shutil
import cv2
import numpy as np
from pathlib import Path

# --- Configuration ---
BASE_DIR = Path(__file__).parent / "rgbd-dataset" / "water_bottle"
DST_DIR = Path(__file__).parent / "water_bottle_yolo"
TRAIN_RATIO = 0.8  # 80-20 split
SEED = 42
CLASS_ID = 0
IMG_W, IMG_H = 640, 480
DEPTH_GLOBAL_MAX = 9870  # pre-computed max across all instances


def get_base_names(src_dir):
    """Extract unique base names and their sequence numbers from any water_bottle_N dir."""
    inst_name = src_dir.name  # e.g. water_bottle_1, water_bottle_10
    pattern = re.compile(rf"^({re.escape(inst_name)}_(\d+)_(\d+))\.png$")
    bases = {}
    for f in os.listdir(src_dir):
        m = pattern.match(f)
        if m:
            bases[m.group(1)] = int(m.group(2))
    return bases


def mask_to_yolo_bbox(mask_path):
    """Convert binary mask to YOLO format bbox string. Returns None if empty."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {mask_path}")

    fg = np.where(mask > 0)
    if len(fg[0]) == 0:
        return None

    y_min, y_max = int(fg[0].min()), int(fg[0].max())
    x_min, x_max = int(fg[1].min()), int(fg[1].max())

    w = x_max - x_min + 1
    h = y_max - y_min + 1
    x_center = (x_min + w / 2) / IMG_W
    y_center = (y_min + h / 2) / IMG_H
    w_norm = w / IMG_W
    h_norm = h / IMG_H

    return f"{CLASS_ID} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


def normalize_depth(depth_path):
    """Read uint16 depth, normalize to uint8 using global max."""
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Depth not found: {depth_path}")
    depth_norm = (depth.astype(np.float32) / DEPTH_GLOBAL_MAX * 255.0)
    return np.clip(depth_norm, 0, 255).astype(np.uint8)


def main():
    # Clean and create output dirs
    if DST_DIR.exists():
        shutil.rmtree(DST_DIR)
    for modality in ["visible", "infrared"]:
        for split in ["train", "test"]:
            (DST_DIR / modality / split).mkdir(parents=True, exist_ok=True)

    # Find all water_bottle instance directories
    instance_dirs = sorted(
        [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("water_bottle_")]
    )
    print(f"Found {len(instance_dirs)} instances: {[d.name for d in instance_dirs]}")

    # Collect all samples across all instances
    all_samples = []  # list of (src_dir, base_name)
    for src_dir in instance_dirs:
        bases = get_base_names(src_dir)
        for base_name in bases:
            all_samples.append((src_dir, base_name))

    # Shuffle and split 80-20
    random.seed(SEED)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * TRAIN_RATIO)
    train_set = set(s[1] for s in all_samples[:split_idx])

    print(f"Total samples: {len(all_samples)}, Train: {split_idx}, Test: {len(all_samples) - split_idx}")

    stats = {"train": 0, "test": 0, "empty_masks": 0}

    for src_dir, base_name in all_samples:
        split = "train" if base_name in train_set else "test"

        rgb_path = src_dir / f"{base_name}.png"
        depth_path = src_dir / f"{base_name}_depth.png"
        mask_path = src_dir / f"{base_name}_mask.png"

        for p in [rgb_path, depth_path, mask_path]:
            if not p.exists():
                print(f"WARNING: Missing {p}, skipping {base_name}")
                break
        else:
            bbox_line = mask_to_yolo_bbox(mask_path)

            label_path = DST_DIR / "visible" / split / f"{base_name}.txt"
            with open(label_path, "w") as f:
                if bbox_line:
                    f.write(bbox_line + "\n")
                else:
                    stats["empty_masks"] += 1

            shutil.copy2(rgb_path, DST_DIR / "visible" / split / f"{base_name}.png")

            depth_norm = normalize_depth(depth_path)
            cv2.imwrite(str(DST_DIR / "infrared" / split / f"{base_name}.png"), depth_norm)

            stats[split] += 1

    print(f"\nConversion complete!")
    print(f"  Train: {stats['train']} images")
    print(f"  Test:  {stats['test']} images")
    print(f"  Empty masks: {stats['empty_masks']}")


if __name__ == "__main__":
    main()
