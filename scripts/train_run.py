import sys
from ultralytics import YOLO, RTDETR

def main():
    name, model, data, batch = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    M = RTDETR if "rtdetr" in model else YOLO
    M(model).train(data=data, epochs=100, imgsz=640, batch=batch, seed=seed,
                   deterministic=True, plots=False,
                   project="/home/daljit/runs_paper", name=name)

if __name__ == "__main__":
    main()
