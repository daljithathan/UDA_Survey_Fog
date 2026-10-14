import sys
from ultralytics import YOLO

DATA = {
    'source_only': '/home/daljit/datasets/cityscapes_yolo/cityscapes_yolo/dataset.yaml',   # clear Cityscapes
    'oracle':      '/home/daljit/datasets/foggy_cs_yolo/foggy_cs_yolo/foggy_cs_fixed.yaml', # Foggy Cityscapes beta=0.02
}

if __name__ == '__main__':
    name = sys.argv[1]                      # 'source_only' or 'oracle'
    model = YOLO('/home/daljit/yolov8s.pt') # COCO-pretrained YOLOv8s
    model.train(data=DATA[name], epochs=100, imgsz=640, batch=32, device=0,
                workers=8, seed=0, deterministic=True,
                project='/home/daljit/runs_paper', name=name, exist_ok=False)
