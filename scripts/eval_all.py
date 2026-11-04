import sys, csv, os
from ultralytics import YOLO, RTDETR

def main():
    home = os.path.expanduser("~")
    keys = ["CS_clear", "FCS_0.005", "FCS_0.01", "FCS_0.02", "FD_test", "FD_all", "RTTS"]
    if len(sys.argv) > 3 and sys.argv[3] == "dcp":
        TESTS = {k: f"{home}/datasets/dcp/{k}.yaml" for k in keys}
    else:
        TESTS = dict(zip(keys, [
            f"{home}/datasets/cityscapes_yolo/cityscapes_yolo/dataset.yaml",
            f"{home}/datasets/fcs_b0.005/data.yaml",
            f"{home}/datasets/fcs_b0.01/data.yaml",
            f"{home}/datasets/fcs_b0.02/data.yaml",
            f"{home}/datasets/fd_yolo/fd_test.yaml",
            f"{home}/datasets/fd_yolo/fd_all.yaml",
            f"{home}/datasets/rtts_yolo/rtts.yaml"]))
    name, weights = sys.argv[1], sys.argv[2]
    Model = RTDETR if "rtdetr" in name.lower() else YOLO
    out = f"{home}/uda-adverse-weather-survey/results/eval_table.csv"
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["row", "testset", "mAP50", "mAP50-95", "classes_evaluated"])
        for k, y in TESTS.items():
            m = Model(weights).val(data=y, imgsz=640, batch=32, split="val", plots=False,
                                   verbose=False, project=f"{home}/eval_runs",
                                   name=f"{name}_{k}", exist_ok=True)
            cls = ",".join(m.names[i] for i in m.box.ap_class_index)
            w.writerow([name, k, round(m.box.map50 * 100, 1), round(m.box.map * 100, 1), cls])
            f.flush()
            print(f"{name:25s} {k:10s} mAP50={m.box.map50*100:.1f}  [{cls}]")

if __name__ == "__main__":
    main()
