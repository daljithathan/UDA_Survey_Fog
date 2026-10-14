from ultralytics import YOLO
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

STEM  = 'munster_000129_000019'
CLEAR = Path('/home/daljit/datasets/cityscapes_yolo/cityscapes_yolo/images/val')
FOGGY = Path('/home/daljit/datasets/foggy_cs_yolo/foggy_cs_yolo/images/val')
model = YOLO('/home/daljit/runs_paper/source_only/weights/last.pt')
NAMES = model.names
COLORS = {0:'#e6194b', 1:'#f58231', 2:'#4363d8', 3:'#911eb4', 4:'#3cb44b', 5:'#f032e6', 6:'#ffe119', 7:'#42d4f4'}
CROP = 0.80                                    # keep top 80% (removes ego-vehicle hood)

fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.25))
used = set()
for ax, (tag, title, d) in zip(axes, [('(a)', 'Clear', CLEAR), ('(b)', 'Foggy (β = 0.02)', FOGGY)]):
    path = d / f'{STEM}_leftImg8bit.png'
    img = Image.open(path).convert('RGB'); W, H = img.size
    r = model.predict(str(path), conf=0.25, imgsz=640, verbose=False, device=0)[0]
    ax.imshow(img.crop((0, 0, W, int(H * CROP))))
    for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, lw=1.0, ec=COLORS[c]))
        used.add(c)
    ax.set_title(f'{tag} {title}: {len(r.boxes)} detections', fontsize=8.5)
    ax.axis('off')
handles = [mpatches.Patch(facecolor='none', edgecolor=COLORS[c], label=NAMES[c]) for c in sorted(used)]
fig.legend(handles=handles, loc='lower center', ncol=len(handles), fontsize=7, frameon=False, bbox_to_anchor=(0.5, -0.02))
plt.tight_layout(rect=(0, 0.07, 1, 1))
plt.savefig('fig1_example.pdf', bbox_inches='tight'); plt.savefig('fig1_example.png', dpi=300, bbox_inches='tight')
print('saved fig1_example.pdf / .png')
