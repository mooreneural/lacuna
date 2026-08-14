"""Figures for the Lacuna software paper.

Every panel is generated from a real run or a real measurement. The K-Ras panels
come from `lacuna discover` on chain A of 4OBE at defaults; the runtime panel
comes from runtime_sweep.py. Nothing here is a schematic of results.

Reuses paper/figures/style.py so the software paper and the methods paper are
typographically the same object.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
REPO = Path(r"C:\Users\clayt\Documents\GitHub\lacuna")
sys.path.insert(0, str(REPO / "paper" / "figures"))
from style import (BLUE, DOUBLE_COL, INK, INK_2, INK_MUTED, ORANGE, AQUA,  # noqa: E402
                   SINGLE_COL, save, use_paper_style)

use_paper_style()
CRYPTIC_FILL = BLUE
OPEN_FILL = "#c9c8c2"


# ── Figure 1: pipeline ───────────────────────────────────────────────────────
def fig_pipeline(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 2.05))
    ax.set_xlim(-1, 101)
    ax.set_ylim(0, 30)
    ax.axis("off")

    stages = ["Input\nPDB / mmCIF",
              "Ensemble\nNMA · MD · Boltz-2\n· external",
              "Per-conformer\ndetection\nalpha-sphere grid",
              "Cross-conformer\nclustering",
              "Learned ranking\n23 features"]
    w, h, gap = 17.2, 10.5, 3.5
    xs = [i * (w + gap) for i in range(len(stages))]
    ymid = 22.5
    for label, x in zip(stages, xs):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, ymid - h / 2), w, h, boxstyle="round,pad=0,rounding_size=1.3",
            linewidth=0.7, edgecolor=INK_MUTED, facecolor="#f6f5f1"))
        ax.text(x + w / 2, ymid, label, ha="center", va="center",
                fontsize=6.7, color=INK, linespacing=1.4)
    for x in xs[:-1]:
        ax.annotate("", xy=(x + w + gap - 0.4, ymid), xytext=(x + w + 0.4, ymid),
                    arrowprops=dict(arrowstyle="-|>", color=INK_2, linewidth=0.85))

    outs = ["ranked JSON", "Boltz YAML\nconstraints", "Vina boxes",
            "pocket PDBs", "crypticity\nscores"]
    ow = 17.2
    oxs = [i * (ow + gap) for i in range(len(outs))]
    for label, x in zip(outs, oxs):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, 1.5), ow, 7.0, boxstyle="round,pad=0,rounding_size=1.1",
            linewidth=0.65, edgecolor=BLUE, facecolor="#eaf2fc"))
        ax.text(x + ow / 2, 5.0, label, ha="center", va="center", fontsize=6.3,
                color=INK_2, linespacing=1.35)

    # One rail: last stage drops onto it, every output hangs from it.
    rail_y = 12.5
    last_c = xs[-1] + w / 2
    ax.plot([last_c, last_c], [ymid - h / 2, rail_y], color=INK_MUTED, linewidth=0.7)
    ax.plot([oxs[0] + ow / 2, last_c], [rail_y, rail_y], color=INK_MUTED, linewidth=0.7)
    for x in oxs:
        ax.annotate("", xy=(x + ow / 2, 8.7), xytext=(x + ow / 2, rail_y),
                    arrowprops=dict(arrowstyle="-|>", color=INK_MUTED, linewidth=0.7))
    ax.text(oxs[0] + ow / 2 + 1.5, rail_y + 1.6, "outputs", ha="left",
            fontsize=6.4, color=INK_MUTED, style="italic")
    save(fig, str(out / "fig2_pipeline"))


# ── Figure 2: the K-Ras worked example ───────────────────────────────────────
def fig_kras(report: Path, out: Path) -> None:
    d = json.loads(report.read_text())
    pockets = d["pockets"]
    n_conf = d["n_conformers"]

    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(DOUBLE_COL, 2.6), gridspec_kw={"width_ratios": [2.0, 1.0]})

    # Rank 1 sits at the top of both panels: y = index, then invert once.
    y = np.arange(len(pockets))
    labels = [str(p["rank"]) for p in pockets]

    # (a) occupancy: which cluster is open in which conformer
    for i, p in enumerate(pockets):
        colour = CRYPTIC_FILL if p["apo_volume_A3"] == 0.0 else OPEN_FILL
        ax0.plot([0, n_conf - 1], [i, i], color="#efeee9", linewidth=0.7,
                 zorder=1, solid_capstyle="round")
        conf = p["appears_in_conformers"]
        ax0.scatter(conf, [i] * len(conf), s=18, color=colour,
                    edgecolor="none", zorder=3)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels)
    ax0.set_ylabel("pocket cluster (rank)")
    ax0.set_xlabel("conformer index (0 = input crystal structure)")
    ax0.set_xlim(-0.9, n_conf - 0.1)
    ax0.set_ylim(-0.9, len(pockets) - 0.1)
    ax0.set_xticks(range(0, n_conf, 5))
    ax0.axvline(0.5, color=INK_MUTED, linewidth=0.6, linestyle=(0, (3, 2)))
    ax0.invert_yaxis()
    ax0.set_title("a   detection is per conformer; clusters span the ensemble",
                  loc="left", fontsize=7.6)
    ax0.legend(handles=[
        mpatches.Patch(color=CRYPTIC_FILL, label="absent in input (cryptic)"),
        mpatches.Patch(color=OPEN_FILL, label="present in input")],
        loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2,
        fontsize=6.4, handlelength=1.1)

    # (b) volume in the input structure against peak volume across the ensemble
    apo = np.array([p["apo_volume_A3"] for p in pockets])
    peak = np.array([p["volume_range_A3"][1] for p in pockets])
    ax1.barh(y, peak, height=0.6, color="#dfe9f6", edgecolor="none",
             label="peak across ensemble")
    ax1.barh(y, apo, height=0.6, color=INK_2, edgecolor="none",
             label="in input structure")
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel("pocket volume (Å$^3$)")
    ax1.set_xlim(0, peak.max() * 1.30)
    ax1.set_ylim(-0.9, len(pockets) - 0.1)
    ax1.invert_yaxis()
    ax1.set_title("b   opening relative to the input", loc="left", fontsize=7.6)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1,
               fontsize=6.4, handlelength=1.1)
    ax1.text(peak[0] * 1.04, 0, "switch-II\n0 → %.0f Å$^3$" % peak[0],
             va="center", ha="left", fontsize=6.2, color=BLUE, linespacing=1.3)
    save(fig, str(out / "fig1_kras"))


# ── Figure 3: runtime ────────────────────────────────────────────────────────
def fig_runtime(path: Path, out: Path) -> None:
    rows = [r for r in json.loads(path.read_text()) if r["ok"]]
    if len(rows) < 5:
        print(f"  runtime: only {len(rows)} rows, skipping")
        return
    n = np.array([r["n_res"] for r in rows], dtype=float)
    t = np.array([r["seconds"] for r in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.3))
    ax.scatter(n, t, s=16, color=BLUE, edgecolor="none", alpha=0.85, zorder=3)
    order = np.argsort(n)
    if len(n) > 6:
        coef = np.polyfit(np.log(n), np.log(t), 1)
        fit = np.exp(np.polyval(coef, np.log(n[order])))
        ax.plot(n[order], fit, color=INK_2, linewidth=1.0,
                linestyle=(0, (4, 2)), zorder=2,
                label=f"slope {coef[0]:.2f} in log-log")
        ax.legend(loc="upper left", fontsize=6.4)
    ax.set_xlabel("chain length (residues)")
    ax.set_ylabel("wall clock (s)")
    ax.set_title(f"20 conformers, NMA backend, single core (n={len(rows)})",
                 loc="left", fontsize=7.4)
    ax.grid(True, alpha=0.5, linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, str(out / "fig4_runtime"))
    print(f"  runtime: n={len(rows)}, median {np.median(t):.1f}s, "
          f"range {t.min():.1f}-{t.max():.1f}s, "
          f"log-log slope {np.polyfit(np.log(n), np.log(t), 1)[0]:.2f}")


# ── Figure 4: recovery across datasets ───────────────────────────────────────
#: Read from README.md's independent-validation table; `learned` is the default
#: ranker, `learned-plm` needs the optional sequence extra.
BENCH = [
    ("CryptoBench\ntest fold", 180, 0.556, 0.661),
    ("PocketMiner", 45, 33 / 45, 36 / 45),
    ("Curated\napo/holo", 22, 10 / 22, 9 / 22),
    ("COACH420\n(non-cryptic)", 144, 125 / 144, None),
]


def fig_benchmarks(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.62, 2.35))
    x = np.arange(len(BENCH))
    w = 0.36
    base = [b[2] for b in BENCH]
    plm = [b[3] for b in BENCH]

    ax.bar(x - w / 2, base, w, color=BLUE, edgecolor="none", label="learned (default)")
    ax.bar([xi + w / 2 for xi, v in zip(x, plm) if v is not None],
           [v for v in plm if v is not None], w, color=ORANGE,
           edgecolor="none", label="learned-plm (optional)")
    for xi, v in zip(x, base):
        ax.text(xi - w / 2, v + 0.018, f"{100*v:.0f}", ha="center",
                fontsize=6.3, color=INK_2)
    for xi, v in zip(x, plm):
        if v is not None:
            ax.text(xi + w / 2, v + 0.018, f"{100*v:.0f}", ha="center",
                    fontsize=6.3, color=INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b[0]}\nn={b[1]}" for b in BENCH], fontsize=6.5)
    ax.set_ylabel("targets recovered in top 5")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v*100:.0f}%"))
    ax.legend(loc="upper left", fontsize=6.4)
    ax.grid(True, axis="y", alpha=0.5, linewidth=0.5)
    ax.set_axisbelow(True)
    save(fig, str(out / "fig3_benchmarks"))


def main() -> None:
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    print("figures ->", out)
    fig_pipeline(out)
    print("  fig2 pipeline")
    fig_kras(HERE / "kras_A" / "pocket_report.json", out)
    print("  fig1 kras")
    rt = HERE / "runtime.json"
    if rt.exists():
        fig_runtime(rt, out)
    else:
        print("  fig4 runtime: runtime.json not ready yet")
    fig_benchmarks(out)
    print("  fig3 benchmarks")


if __name__ == "__main__":
    main()
