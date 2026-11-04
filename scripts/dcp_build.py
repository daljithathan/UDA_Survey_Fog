import glob, os, cv2, numpy as np
from multiprocessing import Pool
H = os.path.expanduser("~")
OUT = f"{H}/datasets/dcp"
NAMES = "nc: 8\nnames: [person, rider, car, truck, bus, train, motorcycle, bicycle]\n"
SETS = {
    "CS_clear":  (f"{H}/datasets/cityscapes_yolo/cityscapes_yolo", "cs", ["val"], "images/val"),
    "FCS_0.005": (f"{H}/datasets/fcs_b0.005", "fcs_b0.005", ["val"], "images/val"),
    "FCS_0.01":  (f"{H}/datasets/fcs_b0.01", "fcs_b0.01", ["val"], "images/val"),
    "FCS_0.02":  (f"{H}/datasets/fcs_b0.02", "fcs_b0.02", ["val"], "images/val"),
    "FD_test":   (f"{H}/datasets/fd_yolo", "fd", ["test", "test_extra"], "images/test"),
    "FD_all":    (f"{H}/datasets/fd_yolo", "fd", ["test", "test_extra"], "[images/test, images/test_extra]"),
    "RTTS":      (f"{H}/datasets/rtts_yolo", "rtts", ["val"], "images/val"),
}

def box(x, r):
    return cv2.boxFilter(x, -1, (r, r))

def guided(I, p, r=60, eps=1e-3):
    mI, mp = box(I, r), box(p, r)
    a = (box(I * p, r) - mI * mp) / (box(I * I, r) - mI * mI + eps)
    b = mp - a * mI
    return box(a, r) * I + box(b, r)

def dehaze(img, w=0.95, t0=0.1, r=15):
    I = img.astype(np.float32) / 255.0
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (r, r))
    dark = cv2.erode(I.min(2), k)
    n = max(1, dark.size // 1000)
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    cand = I.reshape(-1, 3)[idx]
    A = np.maximum(cand[cand.sum(1).argmax()], 1e-3)
    t = 1 - w * cv2.erode((I / A).min(2), k)
    t = guided(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0, t)
    J = (I - A) / np.maximum(t, t0)[..., None] + A
    return (np.clip(J, 0, 1) * 255).astype(np.uint8)

def work(job):
    src, dst = job
    if not os.path.exists(dst):
        cv2.imwrite(dst, dehaze(cv2.imread(src)))

def main():
    jobs, done = [], set()
    for key, (root, out, splits, val) in SETS.items():
        if out not in done:
            for s in splits:
                os.makedirs(f"{OUT}/{out}/images/{s}", exist_ok=True)
                os.makedirs(f"{OUT}/{out}/labels/{s}", exist_ok=True)
                for p in glob.glob(f"{root}/images/{s}/*"):
                    jobs.append((p, f"{OUT}/{out}/images/{s}/{os.path.basename(p)}"))
                for t in glob.glob(f"{root}/labels/{s}/*.txt"):
                    d = f"{OUT}/{out}/labels/{s}/{os.path.basename(t)}"
                    if not os.path.exists(d):
                        os.link(t, d)
            done.add(out)
        open(f"{OUT}/{key}.yaml", "w").write(f"path: {OUT}/{out}\ntrain: {val}\nval: {val}\n" + NAMES)
    print("dehazing", len(jobs), "images", flush=True)
    with Pool(min(16, os.cpu_count())) as pool:
        for i, _ in enumerate(pool.imap_unordered(work, jobs, chunksize=8)):
            if i % 500 == 0:
                print(i, flush=True)
    print("done")

if __name__ == "__main__":
    main()
