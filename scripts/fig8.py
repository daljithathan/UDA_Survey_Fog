import csv, re, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    rows = list(csv.DictReader(open("/home/daljit/uda-adverse-weather-survey/results/eval_table.csv")))
    alias = {"yolov8s_source_only": "yolov8s_source", "yolov8s_oracle": "yolov8s_oracle_b002"}
    g = {}
    for r in rows:
        base = re.sub(r"_s\d+$", "", r["row"])
        base = alias.get(base, base)
        g.setdefault(base, {}).setdefault(r["testset"], []).append(float(r["mAP50"]))

    sets = ["CS_clear", "FCS_0.005", "FCS_0.01", "FCS_0.02", "FD_all", "RTTS"]
    xl = ["Clear", "β=0.005", "β=0.01", "β=0.02", "Foggy\nDriving", "RTTS"]
    lines = [("source",      "Source-only",        "o", "-"),
             ("source_dcp",  "+ DCP at test time", "s", "--"),
             ("source_bn",   "+ BN statistics",    "v", ":"),
             ("oracle_b002", "Labeled fog β=0.02", "^", "-"),
             ("fog_all",     "Labeled fog, all β", "D", "-.")]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, det, title in zip(axes, ["yolov8s", "rtdetr"], ["YOLOv8s", "RT-DETR-L"]):
        for i, (key, lab, mk, ls) in enumerate(lines):
            if det == "rtdetr" and key == "oracle_b002":
                name = "rtdetr_fog_b002"
            else:
                name = f"{det}_{key}"
            d = g.get(name)
            if not d:
                print("missing", name)
                continue
            m = [st.mean(d[s]) for s in sets]
            e = [st.stdev(d[s]) if len(d[s]) > 1 else 0 for s in sets]
            print(f"{title:10s} {lab:20s} n={len(d['CS_clear'])}  " + "  ".join(f"{v:.1f}" for v in m))
            ax.errorbar(range(4), m[:4], yerr=e[:4], marker=mk, ls=ls, capsize=3, label=lab)
            off = (i - 2) * 0.09
            ax.errorbar([4.6 + off, 5.6 + off], m[4:], yerr=e[4:], marker=mk, ls="none",
                        capsize=3, color=ax.lines[-1].get_color())
        ax.axvspan(4.2, 6.0, color="0.92", zorder=0)
        ax.text(5.1, 61, "real fog", ha="center", va="top", fontsize=9)
        ax.set_xticks([0, 1, 2, 3, 4.6, 5.6], xl)
        ax.set_title(title)
        ax.set_xlim(-0.4, 6.0)
        ax.grid(alpha=0.3)

    axes[0].set_ylim(10, 62)
    axes[0].set_ylabel("mAP@0.5 (%)")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out = "/home/daljit/uda-adverse-weather-survey/figures/Figure_8"
    for ext in ["pdf", "png"]:
        fig.savefig(f"{out}.{ext}", dpi=300)
    print("saved", out + ".pdf and .png")

if __name__ == "__main__":
    main()
