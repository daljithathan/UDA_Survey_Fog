import glob, os, re

def main():
    home = os.path.expanduser("~")
    src = f"{home}/datasets/fcs_raw"
    lab = f"{home}/datasets/foggy_cs_yolo/foggy_cs_yolo/labels"
    for beta in ["0.005", "0.01", "0.02"]:
        root = f"{home}/datasets/fcs_b{beta}"
        for split in ["train", "val"]:
            os.makedirs(f"{root}/images/{split}", exist_ok=True)
            os.makedirs(f"{root}/labels/{split}", exist_ok=True)
            for p in glob.glob(f"{src}/**/{split}/**/*_beta_{beta}.png", recursive=True):
                name = re.sub(r"_foggy_beta_[0-9]+\.[0-9]+", "", os.path.basename(p))
                os.link(p, f"{root}/images/{split}/{name}")
            for t in glob.glob(f"{lab}/{split}/*.txt"):
                os.link(t, f"{root}/labels/{split}/{os.path.basename(t)}")
        with open(f"{root}/data.yaml", "w") as f:
            f.write(f"path: {root}\ntrain: images/train\nval: images/val\nnc: 8\n"
                    "names: [person, rider, car, truck, bus, train, motorcycle, bicycle]\n")
        print(beta, {s: (len(os.listdir(f"{root}/images/{s}")), len(os.listdir(f"{root}/labels/{s}")))
                     for s in ["train", "val"]})

if __name__ == "__main__":
    main()
