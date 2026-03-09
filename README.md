# YOLOv11-RGBD — Multi-Architecture RGBD Fusion Framework

A modified Ultralytics codebase supporting **RGBD (RGB + Depth)** object detection across 16 detector architectures with 6 fusion strategies and 7+ benchmark datasets.

## Supported Architectures

| Architecture | Fusion Strategies | Notes |
|---|---|---|
| **YOLOv11** | early, mid, mid-P3, late, score, mid-to-late, share | Most variants: CAS, CTF, PGI, DeepDBB, TransformerFusion, pose/seg/obb tasks |
| **YOLOv12** | early, mid, mid-P3, mid-CTF, late, score, mid-to-late, share | |
| **YOLOv13** | early, mid, mid-P3, late, score, share | Size variants: n/l/x |
| **YOLOv8** | early, mid, mid-P3, mid-CTF, late, score, mid-to-late, share | Includes RGBRGB6C and pose/seg variants |
| **YOLOv9** | early, mid, mid-P3, mid-CTF, late, score, mid-to-late, share | Sizes: t/s/m |
| **YOLOv10** | early, mid, mid-P3, mid-CTF, late, score, mid-to-late, share | Sizes: n/s/m/b/l/x |
| **YOLOv7** | early, mid, mid-P3, mid-CTF, late, score, mid-to-late, share | Also tiny variants |
| **YOLOv6** | early, mid, mid-P3, mid-CTF, late, score, mid-to-late, share | |
| **YOLOv5** | early, mid, mid-P3, mid-CTF, late, score, mid-to-late, share | |
| **YOLOv4** | early, mid, mid-P3, mid-CTF, late, score, mid-to-late, share | Also tiny variants |
| **YOLOv3** | early, mid, mid-P3, mid-CFT, late, score, share | Also tiny variants |
| **YOLOX** | early, mid, late, score, mid-to-late, share | |
| **Hyper-YOLO** | early, mid, mid-CTF, mid-B3 | |
| **PicoDet** | early, mid, mid-CTF, late, score, share | |
| **PP-YOLOE** | early, mid, mid-P3, mid-CTF | |
| **RT-DETR** | early, mid, mid-P3 | ResNet50 backbone |

## Fusion Strategies

| Strategy | Description |
|---|---|
| **Earlyfusion** | Single backbone, 4-channel (RGBD) input |
| **Midfusion** | Dual backbone, features concatenated at P3/P4/P5 |
| **Midfusion-P3** | Dual backbone, features concatenated at P3 only, shared P4+ |
| **Latefusion** | Separate backbones + necks, fused at detection head |
| **Scorefusion** | Independent predictions merged at score level |
| **Mid-to-late fusion** | Features fused progressively from mid to late stages |
| **Share** | Shared-weight backbone for both modalities |

## Datasets

| Dataset | Classes | Description |
|---|---|---|
| **Water Bottle** (1 class) | `water_bottle` | RGB-D Object Dataset (UW), 5,691 samples |
| **FLIR Aligned** (3 classes) | person, car, bicycle | FLIR thermal/visible aligned |
| **KAIST** (1 class) | person | Multispectral pedestrian detection |
| **KAIST8** (1 class) | person | KAIST 8-class subset |
| **LLVIP** (1 class) | person | Low-Light Visible-Infrared Paired |
| **M3FD** (6 classes) | People, Car, Bus, Lamp, Motorcycle, Truck | Multi-spectral detection |
| **VEDAI** (9 classes) | plane, boat, camping_car, car, pick-up, tractor, truck, van, others | Aerial vehicle detection |

Each dataset has modality-specific configs: `-rgbd` (dual-stream), `-vis` (visible only), `-inf` (infrared/depth only).

Model configs are in `ultralytics/cfg/models/<version>-RGBD/` and dataset configs in `ultralytics/cfg/datasets/`.

## Quick Start — Water Bottle Example

### 1. Convert dataset

```bash
python convert_rgbd_to_yolo.py
```

Reads from `rgbd-dataset/water_bottle/`, outputs YOLO-format data to `water_bottle_yolo/` with visible/infrared splits and 80/20 train/test.

### 2. Install dependencies

```bash
pip install -r kaggle_requirements.txt
```

### 3. Update dataset path

Edit `ultralytics/cfg/datasets/water_bottle-rgbd.yaml`:
```yaml
path: /path/to/water_bottle_yolo
```

### 4. Train with COCO pretrained weights (recommended)

```bash
# Earlyfusion (single backbone, 4ch input)
python train_water_bottle_rgbd.py --fusion earlyfusion --device 0 --epochs 100

# Midfusion (dual backbone, concat at P3/P4/P5)
python train_water_bottle_rgbd.py --fusion midfusion --device 0 --epochs 100

# Midfusion-P3 (dual backbone, concat at P3 only)
python train_water_bottle_rgbd.py --fusion midfusion-P3 --device 0 --epochs 100
```

The script handles: template training (1 epoch) → COCO weight transfer → full training.
Use `--skip-template` if the template `.pt` already exists from a previous run.

### 5. Train from scratch

```bash
python train_water_bottle.py
```

## Inference & Tools

```bash
python detect-4C.py              # 4-channel RGBD detection (images/video)
python detect-multispectral.py   # multispectral detection (8-bit or 16-bit, arbitrary channels)
python export.py                 # export model
python val.py                    # validation
python heatmap_RGBD.py           # gradient heatmap visualization
python transform_COCO_to_RGBD.py # transfer COCO weights to RGBD model architectures
```

## Kaggle Notes

- Upload the entire repo as a Kaggle dataset
- Kaggle provides T4 or P100 GPUs — use `--device 0`
- AMP is enabled by default on GPU (disabled automatically on MPS)
