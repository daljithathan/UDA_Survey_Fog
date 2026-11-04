import glob, os
from PIL import Image
home = os.path.expanduser("~")
fd = f"{home}/datasets/Foggy_Driving"
out = f"{home}/datasets/fd_yolo"
MAP = {0: 2, 1: 0, 2: 7, 3: 4, 4: 3, 5: 5, 6: 6, 7: 1}   # FoggyDriving id -> Cityscapes id
for split in ["test", "test_extra"]:
    os.makedirs(f"{out}/images/{split}", exist_ok=True)
    os.makedirs(f"{out}/labels/{split}", exist_ok=True)
    n_img = n_box = 0
    for t in sorted(glob.glob(f"{fd}/bboxGt/{split}/*/*.txt")):
        stem = os.path.basename(t)[:-4]
        rec = os.path.basename(os.path.dirname(t))
        imgs = glob.glob(f"{fd}/leftImg8bit/{split}/{rec}/{stem}_leftImg8bit.*")
        if not imgs:
            print("NO IMAGE for", t); continue
        img = imgs[0]
        W, H = Image.open(img).size
        ext = os.path.splitext(img)[1]
        dst = f"{out}/images/{split}/{stem}{ext}"
        if not os.path.exists(dst):
            os.symlink(img, dst)
        lines = []
        for ln in open(t):
            p = ln.split()
            if len(p) != 5: continue
            c, x1, y1, x2, y2 = int(p[0]), *map(float, p[1:])
            x1, y1 = x1 - 1, y1 - 1                 # 1-based inclusive -> 0-based continuous
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0: continue
            lines.append(f"{MAP[c]} {(x1+w/2)/W:.6f} {(y1+h/2)/H:.6f} {w/W:.6f} {h/H:.6f}")
        open(f"{out}/labels/{split}/{stem}.txt", "w").write("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1; n_box += len(lines)
    print(split, "images:", n_img, "boxes:", n_box)
names = "names: [person, rider, car, truck, bus, train, motorcycle, bicycle]\nnc: 8\n"
open(f"{out}/fd_test.yaml", "w").write(f"path: {out}\ntrain: images/test\nval: images/test\n" + names)
open(f"{out}/fd_all.yaml", "w").write(
    f"path: {out}\ntrain: images/test\nval: [images/test, images/test_extra]\n" + names)
