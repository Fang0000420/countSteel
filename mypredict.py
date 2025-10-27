from ultralytics import YOLO
import os

base_dir = 'base_dir' # 根目录路径
model = YOLO(os.path.join(base_dir, 'runs/detect/train/weights/best.pt'))
model.predict(
    source=os.path.join(base_dir, 'task'),
    imgsz=640,
    save=True,
    conf=0.35,
    show_labels=False,
    line_width=1,
    save_txt=True
    )  