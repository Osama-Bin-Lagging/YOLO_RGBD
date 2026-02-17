# YOLOv11-RGBT Water Bottle Detection — Kaggle Setup

## Dataset

Water bottle data from the [RGB-D Object Dataset](https://rgbd-dataset.cs.washington.edu/) (UW, ICRA 2011). 9 instances (water_bottle_1–7, 9, 10), 5,691 RGBD samples total. Pre-converted to YOLO format in `water_bottle_yolo/` with an 80/20 train/test split (4,552 / 1,139).

## Setup

### 1. Upload to Kaggle

Upload the entire repo as a Kaggle dataset.

### 2. Install dependencies

```bash
pip install -r kaggle_requirements.txt
```

### 3. Update dataset path

Edit `ultralytics/cfg/datasets/water_bottle-rgbt.yaml`:
```yaml
path: /kaggle/working/water_bottle_yolo   # update to your Kaggle path
```

Same for `water_bottle-rgb.yaml` if running the RGB baseline.

## Training

### RGBT with pretrained weights (recommended)

```bash
# Earlyfusion (single backbone, 4ch input)
python train_water_bottle_rgbt.py --fusion earlyfusion --device 0 --epochs 100

# Midfusion (dual backbone, concat at P3/P4/P5)
python train_water_bottle_rgbt.py --fusion midfusion --device 0 --epochs 100

# Midfusion-P3 (dual backbone, concat at P3 only)
python train_water_bottle_rgbt.py --fusion midfusion-P3 --device 0 --epochs 100
```

### RGBT from scratch

```bash
python train_water_bottle.py
```

### RGB baseline

```bash
python train_water_bottle_rgb.py
```

## Existing runs

Previous training results are in `runs/water_bottle/`:
- `wb-yolo11n-RGBT-earlyfusion` — RGBT earlyfusion from scratch
- `wb-yolo11n-RGB-baseline` / `baseline2` — RGB baselines
- `wb-yolo11n-RGB-scratch` — RGB from scratch

Each run contains `weights/best.pt`, `results.csv`, `results.png`, confusion matrices, and validation predictions.

## Inference & tools

```bash
python detect-4C.py          # 4-channel RGBT detection
python detect-multispectral.py  # multispectral detection
python export.py              # export model
python val.py                 # validation
python heatmap_RGBT.py        # gradient heatmap visualization
```

## Notes

- Kaggle provides T4 or P100 GPUs — use `--device 0`
- AMP is enabled by default on GPU (disabled automatically on MPS)
- The unified script handles: template training (1 epoch) → COCO weight transfer → full training
- Use `--skip-template` if the template `.pt` already exists from a previous run
