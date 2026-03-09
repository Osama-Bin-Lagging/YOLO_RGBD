import warnings
warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/11-RGBD/yolo11-RGBD-earlyfusion.yaml")
    model.train(
        data="ultralytics/cfg/datasets/water_bottle-rgbd.yaml",
        cache=False,
        imgsz=640,
        epochs=100,
        batch=8,
        close_mosaic=10,
        workers=2,
        device="mps",
        optimizer="SGD",
        amp=False,
        use_simotm="RGBD",
        channels=4,
        project="runs/water_bottle",
        name="wb-yolo11n-RGBD-earlyfusion",
    )
