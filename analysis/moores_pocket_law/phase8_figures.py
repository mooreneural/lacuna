"""Phase 8: figures A-G. Each is built separately, no subplots.

Uses the repository's existing publication style so these sit alongside the
paper figures rather than looking imported from elsewhere.

    python analysis/moores_pocket_law/phase8_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIG = HERE / "figures"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "paper" / "figures"))

import dataio
from dataio import JACCARD_THRESHOLD, KS, METHODS, boot_ci
from style import (BLUE, ORANGE, AQUA, INK_2, INK_MUTED, SINGLE_COL,
                   TOOL_COLOR, TOOL_LABEL, use_paper_style, save)

SEED = 0
BINS = [(1, 5), (6, 10), (11, 15), (16, 20), (21, 30), (31, 10 ** 6)]


def rows():
    out = []
    with open(HERE / "results.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["total_candidate_count"] = int(r["total_candidate_count"])
            r["qualifying_candidate_exists"] = int(r["qualifying_candidate_exists"])
            r["best_qualifying_rank"] = (int(r["best_qualifying_rank"])
                                         if r["best_qualifying_rank"] not in ("", "None")
                                         else None)
            for k in KS:
                r["hit_at_%d" % k] = int(r["hit_at_%d" % k])
            out.append(r)
    return out


def fig_a(R):
    """Candidate burden vs probability of top-5 conversion."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.25, 2.5))
    cov = [r for r in R if r["qualifying_candidate_exists"]]
    xs, ys, los, his, ns = [], [], [], [], []
    for lo, hi in BINS:
        sel = [r for r in cov if lo <= r["total_candidate_count"] <= hi]
        if len(sel) < 15:
            continue
        m, l, h = boot_ci([r["hit_at_5"] for r in sel], seed=SEED)
        xs.append(np.mean([r["total_candidate_count"] for r in sel]))
        ys.append(m); los.append(m - l); his.append(h - m); ns.append(len(sel))
    ax.errorbar(xs, ys, yerr=[los, his], fmt="o-", color=BLUE, capsize=2.5,
                markersize=4, linewidth=1.4, label="pooled, all methods")
    for x, y, n in zip(xs, ys, ns):
        ax.annotate("n=%d" % n, (x, y), textcoords="offset points",
                    xytext=(0, -13), ha="center", fontsize=5.6, color=INK_MUTED)
    ax.set_xlabel("candidate burden N (bin mean)")
    ax.set_ylabel("P(top-5 hit | site is findable)")
    ax.set_ylim(0.5, 1.03)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%.0f%%" % (100 * v)))
    ax.grid(True, alpha=0.4, linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=6.4, loc="lower left")
    save(fig, str(FIG / "figA_burden_vs_conversion"))


def fig_b_c(R):
    """Budget k vs recovery, and vs conversion of available coverage."""
    cands = dataio.load_candidates()
    ks = np.arange(1, 31)
    for which, fname, ylab in (
            ("recovery", "figB_budget_vs_recovery", "top-k recovery"),
            ("conversion", "figC_budget_vs_conversion",
             "fraction of available coverage converted")):
        fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.25, 2.5))
        for m in METHODS:
            ids = sorted({r["target_id"] for r in R if r["method"] == m})
            jl = {s: cands[s][m] for s in ids}
            cov = np.mean([any(j >= JACCARD_THRESHOLD for j in jl[s]) for s in ids])
            y = [np.mean([any(j >= JACCARD_THRESHOLD for j in jl[s][:k])
                          for s in ids]) for k in ks]
            if which == "conversion":
                y = [v / cov for v in y]
            ax.plot(ks, y, color=TOOL_COLOR[m], linewidth=1.6,
                    label=TOOL_LABEL[m])
            if which == "recovery":
                ax.axhline(cov, color=TOOL_COLOR[m], linewidth=0.7,
                           linestyle=":", alpha=0.75)
        ax.axvline(5, color=INK_MUTED, linewidth=0.7, linestyle="--")
        ax.set_xlabel("candidate budget k")
        ax.set_ylabel(ylab)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%.0f%%" % (100 * v)))
        ax.grid(True, alpha=0.4, linewidth=0.5); ax.set_axisbelow(True)
        ax.legend(fontsize=6.2, loc="lower right")
        save(fig, str(FIG / fname))


def fig_d(R):
    """Detection failure vs ranking failure at k=5."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.25, 2.4))
    x = np.arange(len(METHODS))
    hit, rank, det = [], [], []
    for m in METHODS:
        sub = [r for r in R if r["method"] == m]
        n = len(sub)
        hit.append(sum(r["hit_at_5"] for r in sub) / n)
        rank.append(sum(1 for r in sub if r["qualifying_candidate_exists"]
                        and not r["hit_at_5"]) / n)
        det.append(sum(1 for r in sub
                       if not r["qualifying_candidate_exists"]) / n)
    ax.bar(x, hit, 0.62, color=BLUE, label="recovered in top 5")
    ax.bar(x, rank, 0.62, bottom=hit, color="#c9c8c2",
           label="found, out-ranked")
    ax.bar(x, det, 0.62, bottom=np.array(hit) + np.array(rank),
           color="#ebeae4", label="never proposed")
    for xi, (h, rk) in enumerate(zip(hit, rank)):
        ax.text(xi, h / 2, "%.0f%%" % (100 * h), ha="center", va="center",
                fontsize=6.4, color="white")
        ax.text(xi, h + rk / 2, "%.0f%%" % (100 * rk), ha="center",
                va="center", fontsize=6.2, color=INK_2)
    ax.set_xticks(x); ax.set_xticklabels([TOOL_LABEL[m] for m in METHODS],
                                         fontsize=6.8)
    ax.set_ylabel("share of targets")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "%.0f%%" % (100 * v)))
    ax.legend(fontsize=6.2, loc="upper center", ncol=3,
              bbox_to_anchor=(0.5, 1.22), frameon=False)
    save(fig, str(FIG / "figD_failure_composition"))


def fig_e(R):
    """Depth of the first qualifying candidate vs burden."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.25, 2.5))
    rng = np.random.default_rng(SEED)
    for m in METHODS:
        sub = [r for r in R if r["qualifying_candidate_exists"]
               and r["method"] == m]
        x = np.array([r["total_candidate_count"] for r in sub], float)
        y = np.array([r["best_qualifying_rank"] for r in sub], float)
        j = rng.uniform(-0.28, 0.28, size=len(x))
        ax.scatter(x + j, y + j, s=3.2, alpha=0.22, color=TOOL_COLOR[m],
                   edgecolors="none", label=TOOL_LABEL[m])
    ax.axhline(5, color=INK_MUTED, linestyle="--", linewidth=0.8)
    ax.text(62, 5.6, "top-5 budget", fontsize=6, color=INK_2, ha="right")
    ax.set_xlabel("total candidate count N")
    ax.set_ylabel("rank of first qualifying candidate")
    ax.set_yscale("log"); ax.set_xlim(0, 65)
    ax.grid(True, alpha=0.35, linewidth=0.5); ax.set_axisbelow(True)
    leg = ax.legend(fontsize=6.2, loc="upper left", markerscale=3)
    for lh in leg.legend_handles:
        lh.set_alpha(1)
    save(fig, str(FIG / "figE_rank_vs_burden"))


def fig_f(R):
    """Marginal probability the first qualifying candidate lands at rank r."""
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.25, 2.4))
    for m in METHODS:
        sub = [r for r in R if r["method"] == m]
        n = len(sub)
        hist = np.zeros(21)
        for r in sub:
            rk = r["best_qualifying_rank"]
            if rk is not None and rk <= 20:
                hist[rk] += 1
        ax.plot(np.arange(1, 21), hist[1:] / n, "o-", markersize=2.8,
                linewidth=1.3, color=TOOL_COLOR[m], label=TOOL_LABEL[m])
    ax.set_xlabel("rank r")
    ax.set_ylabel("P(first qualifying candidate at r)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.4, linewidth=0.5); ax.set_axisbelow(True)
    ax.legend(fontsize=6.2)
    save(fig, str(FIG / "figF_marginal_yield"))


def fig_g():
    """The within-target sweep: the causal test, and eta."""
    sweep = json.loads((HERE / "phase6_sweep.json").read_text())
    conf = json.loads((HERE / "phase5_confounders.json").read_text())

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, 2.5))
    steps = list(conf["efficiency"])
    xs = np.arange(len(steps))
    eta = [conf["efficiency"][s]["eta"] for s in steps]
    lo = [conf["efficiency"][s]["eta"] - conf["efficiency"][s]["eta_ci"][0]
          for s in steps]
    hi = [conf["efficiency"][s]["eta_ci"][1] - conf["efficiency"][s]["eta"]
          for s in steps]
    ax.errorbar(xs, eta, yerr=[lo, hi], fmt="o", color=ORANGE, capsize=3,
                markersize=5, linewidth=1.4)
    ax.axhline(1.0, color=INK_MUTED, linestyle=":", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([s.replace("_to_", " to ") for s in steps], fontsize=6)
    ax.set_xlabel("conformer-count step (burden increases left to right)")
    ax.set_ylabel(r"$\eta = \Delta R_5 / \Delta C$")
    ax.grid(True, alpha=0.4, linewidth=0.5); ax.set_axisbelow(True)
    ax.set_title("efficiency does not decline monotonically",
                 fontsize=7.5, color=INK_2)
    save(fig, str(FIG / "figG_efficiency_by_burden"))

    # The decisive within-target panel.
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, 2.5))
    ks = [1, 3, 5, 10, 20]
    d = [100 * sweep["covered_at_both"]["hit_at_%d" % k]["delta"] for k in ks]
    dl = [100 * (sweep["covered_at_both"]["hit_at_%d" % k]["delta"]
                 - sweep["covered_at_both"]["hit_at_%d" % k]["delta_ci"][0])
          for k in ks]
    dh = [100 * (sweep["covered_at_both"]["hit_at_%d" % k]["delta_ci"][1]
                 - sweep["covered_at_both"]["hit_at_%d" % k]["delta"])
          for k in ks]
    ax.errorbar(range(len(ks)), d, yerr=[dl, dh], fmt="s", color=BLUE,
                capsize=3, markersize=5, linewidth=1.4)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(range(len(ks))); ax.set_xticklabels(ks)
    ax.set_xlabel("budget k")
    ax.set_ylabel("change in hit rate (points)")
    ax.set_title("same targets, burden 7.1 to 22.8 candidates (n=%d)"
                 % sweep["covered_at_both"]["hit_at_5"]["n"],
                 fontsize=7.5, color=INK_2)
    ax.grid(True, alpha=0.4, linewidth=0.5); ax.set_axisbelow(True)
    save(fig, str(FIG / "figG2_within_target_effect"))


def main() -> None:
    use_paper_style()
    FIG.mkdir(exist_ok=True)
    R = rows()
    fig_a(R); print("  figA burden vs conversion")
    fig_b_c(R); print("  figB/figC budget curves")
    fig_d(R); print("  figD failure composition")
    fig_e(R); print("  figE rank vs burden")
    fig_f(R); print("  figF marginal yield")
    fig_g(); print("  figG/figG2 efficiency and within-target effect")
    print("figures -> %s" % FIG)


if __name__ == "__main__":
    main()
