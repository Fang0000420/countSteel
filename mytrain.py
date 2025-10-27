from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(
    data='D:\pyLearn\countSteel\steel.yaml',
    epochs=100,
    imgsz=640,
    batch=16
    )  