from ultralytics import YOLO
import os

base_dir = 'base_dir'
model = YOLO(os.path.join(base_dir, 'yolo11n.pt'))
model.train(
    data=os.path.join(base_dir, 'steel.yaml'),
    epochs=100,
    imgsz=640,
    batch=16
    )  