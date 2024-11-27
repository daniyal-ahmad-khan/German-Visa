from ultralytics import YOLO

modelx = YOLO("yolov8n.yaml")
results = modelx.train(data="data.yaml", epochs=500, imgsz=640,amp=False)
