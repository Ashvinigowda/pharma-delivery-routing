#!/usr/bin/env python3
"""Aggregate results/raw into tables (results/*.md) and figures (figures/*.png)."""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiments import BKS, INSTANCES, ROOT, get_instance
from vrptw.stats import mann_whitney_p, mean, std

RAW = os.path.join(ROOT, "results", "raw")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)


def load(prefix):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RAW, prefix + "*.json")))]


def group(runs, key):
    g = {}
    for r in runs:
        g.setdefault(key(r), []).append(r)
    return g


def best_at(history, t):
    b = history[0][2]
    for _, tt, c in history:
        if tt <= t:
            b = c
    return b


def main_tables():
    base = json.load(open(os.path.join(ROOT, "results", "baselines.json")))
    runs = load("main_")
    g = group(runs, lambda r: (r["instance"], "memetic" if r["params"].get("memetic", True) else "plain"))
    lines = ["| Instance | BKS | NN | CW | Plain EA mean | Memetic best | Memetic mean | Memetic std | Gap% (best) | Gap% (mean) | Veh | p-value |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    csv = ["instance,bks,nn,cw,plain_mean,plain_best,memetic_best,memetic_mean,memetic_std,gap_best,gap_mean,vehicles,p_value,runs,all_valid"]
    for n in INSTANCES:
        if (n, "memetic") not in g:
            continue
        m = [r["distance"] for r in g[(n, "memetic")]]
        p = [r["distance"] for r in g.get((n, "plain"), [])]
        bks = BKS.get(n, (min(m), 0))[0]
        pv = mann_whitney_p(m, p) if p else float("nan")
        veh = min(g[(n, "memetic")], key=lambda r: r["distance"])["vehicles"]
        valid = all(r["valid"] for r in g[(n, "memetic")] + g.get((n, "plain"), []))
        gb, gm = [round(100 * (x - bks) / bks, 2) + 0.0 for x in (min(m), mean(m))]
        bk = f"{bks:.2f}" if n in BKS else f"{min(m):.2f}*"
        lines.append(f"| {n} | {bk} | {base[n]['NN']['distance']:.2f} | {base[n]['CW']['distance']:.2f} | "
                     f"{mean(p):.2f} | {min(m):.2f} | {mean(m):.2f} | {std(m):.2f} | {gb:+.2f} | {gm:+.2f} | {veh} | {pv:.4f} |")
        csv.append(",".join(str(x) for x in (n, bks, base[n]["NN"]["distance"], base[n]["CW"]["distance"], mean(p), min(p),
                                              min(m), mean(m), std(m), gb, gm, veh, pv, len(m), valid)))
    note = ("\n\\* PHARMA60 is synthetic; the value shown is the best solution found by our own runs, not a literature value.\n"
            "BKS minimise vehicles first, then distance, so a distance-only solver can legitimately go below them "
            "(negative gap) by using more vehicles.\n")
    open(os.path.join(ROOT, "results", "main_results.md"), "w").write("\n".join(lines) + "\n" + note)
    open(os.path.join(ROOT, "results", "main_results.csv"), "w").write("\n".join(csv) + "\n")
    return g


def convergence(g):
    names = [n for n in ("C101", "R101", "RC101", "R201") if (n, "memetic") in g]
    if not names:
        return
    fig, axs = plt.subplots(1, len(names), figsize=(4.2 * len(names), 3.4))
    axs = [axs] if len(names) == 1 else axs
    for ax, n in zip(axs, names):
        for v, col in (("memetic", "tab:red"), ("plain", "tab:blue")):
            rs = g[(n, v)]
            T = max(r["history"][-1][1] for r in rs)
            grid = [T * i / 100 for i in range(101)]
            ys = [mean([best_at(r["history"], t) for r in rs]) for t in grid]
            ax.plot(grid, ys, color=col, label=v + " EA")
        ax.set_title(n)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("mean best distance")
        ax.grid(alpha=0.3)
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "convergence.png"), dpi=150)
    plt.close(fig)


def boxplots(g):
    names = [n for n in INSTANCES if (n, "memetic") in g]
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))
    for ax, n in zip(axs.flat, names):
        ax.boxplot([[r["distance"] for r in g[(n, "plain")]], [r["distance"] for r in g[(n, "memetic")]]],
                   tick_labels=["plain", "memetic"])
        ax.set_title(n)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "boxplots.png"), dpi=130)
    plt.close(fig)


def route_maps(g):
    for n in ("C101", "R101", "RC101", "PHARMA60"):
        if (n, "memetic") not in g:
            continue
        inst = get_instance(n)
        best = min(g[(n, "memetic")], key=lambda r: r["distance"])
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.get_cmap("tab20")
        for k, r in enumerate(best["routes"]):
            pts = [0] + r + [0]
            ax.plot([inst.x[i] for i in pts], [inst.y[i] for i in pts], "-", color=cmap(k % 20), lw=1.2)
        ax.scatter(inst.x[1:], inst.y[1:], s=12, c="k", zorder=3)
        ax.scatter([inst.x[0]], [inst.y[0]], marker="s", s=70, c="red", zorder=4, label="depot")
        ax.set_title(f"{n}: distance {best['distance']:.2f}, {len(best['routes'])} vehicles")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(FIG, f"routes_{n}.png"), dpi=130)
        plt.close(fig)


def param_tables():
    runs = load("param_")
    if not runs:
        return
    g = group(runs, lambda r: r["tag"].rsplit("_", 1)[0].split("_", 2)[2])
    inst = runs[0]["instance"]
    lines = [f"Parameter study on {inst} (distance over independent runs; differences are small relative to run-to-run noise)\n",
             "| Configuration | Mean | Std | Best | Runs |", "|---|---|---|---|---|"]
    for k, rs in g.items():
        d = [r["distance"] for r in rs]
        lines.append(f"| {k} | {mean(d):.2f} | {std(d):.2f} | {min(d):.2f} | {len(d)} |")
    open(os.path.join(ROOT, "results", "param_study.md"), "w").write("\n".join(lines) + "\n")


def fleet_table():
    runs = load("fleet_")
    if not runs:
        return
    g = group(runs, lambda r: r["params"]["fixed"])
    lines = ["| Fixed cost per vehicle | Mean distance | Mean vehicles | Min vehicles |", "|---|---|---|---|"]
    xs, ds, vs = [], [], []
    for f in sorted(g):
        d = [r["distance"] for r in g[f]]
        v = [r["vehicles"] for r in g[f]]
        lines.append(f"| {f:.0f} | {mean(d):.2f} | {mean(v):.1f} | {min(v)} |")
        xs.append(f); ds.append(mean(d)); vs.append(mean(v))
    open(os.path.join(ROOT, "results", "fleet_study.md"), "w").write("\n".join(lines) + "\n")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(vs, ds, "o-")
    for x, v, d in zip(xs, vs, ds):
        ax.annotate(f"F={x:.0f}", (v, d), textcoords="offset points", xytext=(4, 4), fontsize=8)
    ax.set_xlabel("vehicles used")
    ax.set_ylabel("total distance")
    ax.set_title("Distance vs fleet size trade-off")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fleet_tradeoff.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    g = main_tables()
    convergence(g)
    boxplots(g)
    route_maps(g)
    param_tables()
    fleet_table()
    print("tables -> results/*.md, figures -> figures/*.png")
