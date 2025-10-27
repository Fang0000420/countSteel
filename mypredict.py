from ultralytics import YOLO

model = YOLO('D:/pyLearn/countSteel/runs/detect/train2/weights/best.pt')
model.predict(
    source='D:/pyLearn/countSteel/task',
    imgsz=640,
    save=True,
    conf=0.35,
    show_labels=False,
    line_width=1,
    save_txt=True
    )  