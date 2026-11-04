import csv, re, statistics as st

def main():
    rows = list(csv.DictReader(open("/home/daljit/uda-adverse-weather-survey/results/eval_table.csv")))
    sets = ["CS_clear", "FCS_0.005", "FCS_0.01", "FCS_0.02", "FD_test", "FD_all", "RTTS"]
    alias = {"yolov8s_source_only": "yolov8s_source", "yolov8s_oracle": "yolov8s_oracle_b002"}
    groups = {}
    for r in rows:
        base = re.sub(r"_s\d+$", "", r["row"])
        base = alias.get(base, base)
        groups.setdefault(base, {}).setdefault(r["testset"], []).append(float(r["mAP50"]))
    print(f"{'model':26s}{'n':>3s}" + "".join(f"{s:>11s}" for s in sets))
    for g, d in groups.items():
        n = max(len(v) for v in d.values())
        cells = []
        for s in sets:
            v = d.get(s, [])
            cells.append("-" if not v else f"{st.mean(v):.1f}" + (f"±{st.stdev(v):.1f}" if len(v) > 1 else ""))
        print(f"{g:26s}{n:>3d}" + "".join(f"{c:>11s}" for c in cells))

if __name__ == "__main__":
    main()
