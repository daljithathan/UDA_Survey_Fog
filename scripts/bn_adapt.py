import sys, glob, cv2, torch, numpy as np
from ultralytics import YOLO, RTDETR

def load(p, size=640):
    im = cv2.imread(p)
    h, w = im.shape[:2]
    r = size / max(h, w)
    nw, nh = round(w * r), round(h * r)
    im = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)
    ph, pw = (32 - nh % 32) % 32, (32 - nw % 32) % 32
    im = cv2.copyMakeBorder(im, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2,
                            cv2.BORDER_CONSTANT, value=(114, 114, 114))
    im = im[:, :, ::-1] / 255.0
    return torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float()

def main():
    src, tdir, out = sys.argv[1:4]
    m = RTDETR(src) if "rtdetr" in src else YOLO(src)
    net = m.model.cuda().float().eval()
    for mod in net.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.reset_running_stats(); mod.momentum = None; mod.train()
    imgs = sorted(glob.glob(f"{tdir}/*.png") + glob.glob(f"{tdir}/*.jpg"))
    with torch.no_grad():
        for i in range(0, len(imgs), 32):
            net(torch.stack([load(p) for p in imgs[i:i+32]]).cuda())
    net.eval(); m.save(out); print("saved", out, "from", len(imgs), "images")

if __name__ == "__main__":
    main()
