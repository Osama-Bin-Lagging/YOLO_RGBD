# YOLOv11-RGBD — Multi-Architecture RGBD Fusion Framework

A modified Ultralytics codebase supporting **RGBD (RGB + Depth)** object detection across 16 detector architectures with 6 fusion strategies and 7+ benchmark datasets.

## Overview

YOLO is one of the most popular real-time object detection models — feed it a photo and it draws boxes around the things in it. Standard YOLO only looks at **color images** (RGB: red, green, blue channels). This project is a modified version that looks at **color images and depth maps at the same time**.

A depth map is a grayscale image where each pixel's brightness represents how far that point is from the camera, the way an iPhone LiDAR scan or a Kinect captures the world. The two modalities fail in different ways: color cameras get confused by shadows, low light, and objects that look similar to their background; depth sensors don't care about lighting but can't tell a red apple from a green one. Used together, the two signals cover for each other's blind spots, so the model tends to be more accurate — especially in cluttered scenes, low light, or when objects are camouflaged against similar-colored backgrounds.

You give it matched pairs of images — an RGB photo and a depth photo of the exact same moment — and it learns to detect whatever objects you train it on. The repo ships with a small water-bottle example and configs for standard research benchmarks: pedestrians (KAIST, LLVIP), vehicles (M3FD, VEDAI), and thermal/visible aligned scenes (FLIR). Output is the same as normal YOLO: bounding boxes plus class labels.

## Architectural Changes from Standard YOLO

Standard YOLO is built around 3-channel RGB input running through a single backbone (the network that turns pixels into features). To make it handle RGB + Depth, several real architectural changes were made:

1. **Input widened from 3 to 4 channels.** The 4th channel carries depth. This sounds trivial but it cascades through everything — the first convolution layer, the data loader, the pretrained weights, and the augmentation pipeline.

2. **A channel splitter (`Silence` + `SilenceChannel`) was inserted at the very start of every model.** The data loader stacks RGB and depth into one 4-channel tensor (so all of YOLO's normal batching, augmentation, and resizing code keeps working unchanged), and a custom layer immediately separates them again inside the model. This is the trick that lets the project layer multi-modal architectures on top of a mostly untouched data pipeline.

3. **Six different fusion strategies were built**, each defining *where* in the network the color and depth signals merge. This is the actual research contribution:
   - **Early fusion** glues them together at the input and uses one backbone — fastest and smallest, but the model has to figure out depth-vs-color on its own.
   - **Mid fusion** runs two parallel backbones, one for RGB and one for depth, and concatenates their feature maps deeper in the network — roughly double the backbone parameters, but each modality gets its own dedicated feature extractor.
   - **Late fusion** keeps the two streams entirely separate until the end and merges only the predictions.
   - **Score fusion, mid-to-late, and share** are intermediate variations trading parameter count against how strongly the streams influence each other.

   The point of having all six is to be able to ask: *where in the network is the best place to inject depth?* That answer turns out to depend on the dataset and the task.

4. **A custom 3-step training pipeline** was added to handle pretrained weights. Pretrained YOLO checkpoints (from COCO) are 3-channel; the new model has 4-channel (or even split 3+1) inputs. So the new training script: (a) builds the new model with random weights and runs one epoch just to "shape" it, (b) walks through the COCO checkpoint and copies over every layer that's compatible while adapting shapes where they aren't, then (c) starts real training from that grafted starting point. Without this, the new model would have to learn from scratch and would need huge amounts of training data to get anywhere.

5. **A data converter** (`convert_rgbd_to_yolo.py`) turns standard RGB-D datasets — where depth is stored as 16-bit millimeter values — into the 8-bit grayscale format YOLO's pipeline can consume. It also generates bounding boxes from the segmentation masks the original datasets ship with.

6. **The whole framework was extended to 16 detector families**: every major YOLO version from v3 through v13, plus YOLOX, PicoDet, PP-YOLOE, RT-DETR, and Hyper-YOLO. Each one has its own set of fusion configs, so you can pick the speed/accuracy trade-off that fits your hardware and still ask the same "where should depth be fused?" question.

## Why It Matters

Object detection has been a "color only" field for years, mostly out of habit. Cheap depth sensors (iPhone LiDAR, Intel RealSense, stereo cameras on robots) have changed that, and there's now a lot of unused RGB+D data sitting around. This project is essentially a research toolbox for systematically figuring out the best way to use it — without having to rewrite each YOLO variant from scratch.

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
