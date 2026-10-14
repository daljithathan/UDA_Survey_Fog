from ultralytics import YOLO
from pathlib import Path
CLEAR = Path('/home/daljit/datasets/cityscapes_yolo/cityscapes_yolo/images/val')
FOGGY = Path('/home/daljit/datasets/foggy_cs_yolo/foggy_cs_yolo/images/val')
LBL   = Path('/home/daljit/datasets/cityscapes_yolo/cityscapes_yolo/labels/val')
model = YOLO('/home/daljit/runs_paper/source_only/weights/last.pt')
stems = sorted(p.stem for p in CLEAR.glob('*.png'))
def count(folder):
    return {Path(r.path).stem: len(r.boxes) for r in
            model.predict([str(folder/f'{s}.png') for s in stems], conf=0.25, imgsz=640,
                          stream=True, verbose=False, device=0)}
c, f = count(CLEAR), count(FOGGY)
rows = []
for s in stems:
    gt = len(set((LBL/f'{s}.txt').read_text().split('\n')) - {''})
    if 10 <= gt <= 40:                       # readable scenes only
        rows.append((c[s] - f[s], s, gt, c[s], f[s]))
rows.sort(reverse=True)
print(f"{'stem':45s} GT  clear foggy  drop")
for d, s, gt, cc, ff in rows[:10]:
    print(f"{s:45s} {gt:3d} {cc:5d} {ff:5d} {d:5d}")
