import argparse
import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args()

    model = YOLO("yolo11n.pt")  # standard 3-channel pretrained YOLOv11 nano
    model.train(
        data="ultralytics/cfg/datasets/water_bottle-rgb.yaml",
        cache=False,
        imgsz=640,
        epochs=args.epochs,
        batch=8,
        close_mosaic=10,
        workers=2,
        device=args.device,
        optimizer="SGD",
        amp=args.device != "mps",
        use_simotm="BGR",
        channels=3,
        project="runs/water_bottle",
        name="wb-yolo11n-RGB-baseline",
    )
