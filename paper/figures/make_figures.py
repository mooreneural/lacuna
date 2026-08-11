#!/usr/bin/env python3
"""Build the paper's data figures from paper/data/analysis.json.

Every number is read from that file; nothing is computed here. Figures and text
therefore cannot drift apart, and regenerating after a re-analysis is one command.

    python paper/figures/make_figures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import NullFormatter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from style import (  # noqa: E402
    DOUBLE_COL, INK, INK_2, INK_MUTED, LOST_FILL, MISSING_FILL, SINGLE_COL,
    SURFACE, TOOL_COLOR, TOOL_LABEL, TOOL_ORDER, UNION, pct_axis, save,
    use_paper_style,
)

DATA = HERE.parent / "data" / "analysis.json"


def fig_composition(d, out):
    """Where each detector's structures go: recovered, out-ranked, or never found.

    A stacked bar, because the question is composition of a whole: every structure
    lands in exactly one of three states. Recovered carries the tool's colour;
    the two failure modes are greys, so absence never competes for identity.
    """
    t = d["splits"]["test"]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.62, 2.05))
    ys = np.arange(len(TOOL_ORDER))

    for i, tool in enumerate(TOOL_ORDER):
        e = t["tools"][tool]
        seg = [e["topk"]["5"][0], e["lost_to_ranking"], e["never_proposed"]]
        fills = [TOOL_COLOR[tool], LOST_FILL, MISSING_FILL]
        left = 0.0
        for val, fill in zip(seg, fills):
            ax.barh(ys[i], val, left=left, height=0.6, color=fill,
                    edgecolor=SURFACE, linewidth=1.4)
            left += val
        # Direct-label the two quantities the paper argues about. A segment
        # narrower than its own label gets the label above it instead of inside,
        # which is the only way P2Rank's 3% stays legible.
        ax.text(seg[0] / 2, ys[i], f"{seg[0]:.0%}", ha="center", va="center",
                color=SURFACE if seg[0] > 0.12 else INK, fontsize=7.5,
                fontweight="bold")
        if seg[1] >= 0.08:
            ax.text(seg[0] + seg[1] / 2, ys[i], f"{seg[1]:.0%}", ha="center",
                    va="center", color=INK_2, fontsize=7)
        else:
            ax.text(seg[0] + seg[1] / 2, ys[i] - 0.40, f"{seg[1]:.0%}",
                    ha="center", va="bottom", color=INK_2, fontsize=6.8)
        ax.text(1.012, ys[i], f"{e['n_candidates_mean']:.0f}", ha="left",
                va="center", color=INK_2, fontsize=7)

    ax.text(1.012, -0.62, "candidates", ha="left", va="center",
            color=INK_MUTED, fontsize=6.5)
    ax.set_yticks(ys, [TOOL_LABEL[t_] for t_ in TOOL_ORDER])
    ax.invert_yaxis()
    pct_axis(ax, "x")
    ax.set_xlabel("Structures")
    ax.xaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)

    # The recovered segment is drawn in each tool's own colour, so its legend key
    # is the three hues side by side rather than one grey chip that belongs to
    # nothing on the plot.
    recovered_key = tuple(
        plt.Rectangle((0, 0), 1, 1, fc=TOOL_COLOR[t_], ec=SURFACE)
        for t_ in TOOL_ORDER
    )
    handles = [
        recovered_key,
        plt.Rectangle((0, 0), 1, 1, fc=LOST_FILL, ec=SURFACE),
        plt.Rectangle((0, 0), 1, 1, fc=MISSING_FILL, ec=SURFACE),
    ]
    ax.legend(handles,
              ["Recovered in top 5", "Found, but out-ranked", "Never proposed"],
              loc="upper center", bbox_to_anchor=(0.5, -0.34), ncol=3,
              handlelength=1.6, columnspacing=1.3,
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    save(fig, str(out / "fig2_composition"))


def fig_union(d, out):
    """Coverage as detectors are combined."""
    t = d["splits"]["test"]
    combos = [
        ("p2rank", "P2Rank"),
        ("ifsitepred", "IF-Site\nPred"),
        ("lacuna", "Lacuna"),
        ("fpocket", "fpocket"),
        ("fpocket+ifsitepred", "fpocket\n+ IF-SitePred"),
        ("fpocket+p2rank+ifsitepred+lacuna", "all four"),
    ]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.66, 2.1))
    xs = np.arange(len(combos))
    for i, (key, _) in enumerate(combos):
        m, lo, hi = t["union_oracle"][key]
        single = "+" not in key
        ax.bar(xs[i], m, width=0.62,
               color=TOOL_COLOR[key] if single else UNION,
               edgecolor=SURFACE, linewidth=1.2)
        ax.errorbar(xs[i], m, yerr=[[m - lo], [hi - m]], fmt="none",
                    ecolor=INK_2, elinewidth=0.9, capsize=2.2)
        ax.text(xs[i], hi + 0.025, f"{m:.0%}", ha="center", va="bottom",
                color=INK, fontsize=7)

    ax.set_xticks(xs, [lab for _, lab in combos])
    pct_axis(ax, "y", upper=1.0)
    ax.set_ylabel("Structures with the site among\nany proposed candidate")
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    ceiling = 1 - t["missed_by_all"]
    ax.axhline(ceiling, color=INK_MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(-0.45, ceiling + 0.018,
            f"{t['missed_by_all']:.1%} invisible to every detector",
            ha="left", va="bottom", color=INK_2, fontsize=6.6)
    ax.set_ylim(0, 1.16)
    save(fig, str(out / "fig3_union"))


def fig_budget(d, out):
    """Recovery against how many candidates a user is willing to examine.

    A line chart: the reader's question is how the quantity moves with budget, and
    the crossover between the single tools and the consensus is the whole point.
    """
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.35), sharey=True)
    for ax, split in zip(axes, ("test", "train")):
        s = d["splits"][split]
        budgets = sorted(int(b) for b in s["budget"])
        for tool in TOOL_ORDER:
            ys = [s["budget"][str(b)]["single"][tool][0] for b in budgets]
            ax.plot(budgets, ys, color=TOOL_COLOR[tool], marker="o", ms=3.4,
                    label=TOOL_LABEL[tool], zorder=3)
        ys = [s["budget"][str(b)]["union"][0] for b in budgets]
        ax.plot(budgets, ys, color=UNION, marker="s", ms=3.4, ls=(0, (5, 2)),
                label="Consensus of all four", zorder=4)

        ax.set_xscale("log")
        ax.set_xticks(budgets, [str(b) for b in budgets])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis="x", which="minor", length=0)
        ax.set_xlabel("Candidates examined per structure")
        ax.set_title(f"{'Designated test fold' if split == 'test' else 'Train folds'}"
                     f"  (n={s['n']})", color=INK_2, pad=4)
        ax.grid(True, alpha=0.7)
        ax.set_axisbelow(True)
    pct_axis(axes[0], "y", upper=1.0)
    axes[0].set_ylabel("Recovery")
    axes[0].legend(loc="lower right", handlelength=1.6)
    save(fig, str(out / "fig4_budget"))


def fig_headroom(d, out):
    """What the achievable ceiling is, and which part of it ranking can reach."""
    h = d["splits"]["test"]["headroom"]
    steps = [
        ("Achieved\n(best tool, top 5)", h["achieved_top5"], TOOL_COLOR["lacuna"]),
        ("+ perfect ranking\nof one tool", h["perfect_ranking_one_tool"], LOST_FILL),
        ("+ perfect ranking\nof all four", h["perfect_ranking_union"], UNION),
    ]
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.1))
    xs = np.arange(len(steps))
    prev = 0.0
    for i, (lab, val, fill) in enumerate(steps):
        ax.bar(xs[i], val - prev, bottom=prev, width=0.6, color=fill,
               edgecolor=SURFACE, linewidth=1.2)
        if i:
            ax.text(xs[i], val + 0.02, f"+{(val - prev) * 100:.1f} pts",
                    ha="center", va="bottom", color=INK_2, fontsize=7)
            # Waterfall connector: makes each step read as continuing the last.
            ax.plot([xs[i - 1] + 0.3, xs[i] - 0.3], [prev, prev],
                    color=INK_MUTED, lw=0.7, ls=(0, (2, 2)), zorder=1)
        else:
            ax.text(xs[i], val + 0.02, f"{val:.0%}", ha="center", va="bottom",
                    color=INK, fontsize=7)
        prev = val
    ax.axhline(1 - h["irreducible"], color=INK_MUTED, lw=0.8, ls=(0, (4, 3)))
    ax.text(-0.42, 1 - h["irreducible"] + 0.015,
            f"ceiling {1 - h['irreducible']:.0%}", ha="left", va="bottom",
            color=INK_2, fontsize=6.8)
    ax.set_xticks(xs, [s[0] for s in steps])
    pct_axis(ax, "y", upper=1.05)
    ax.set_ylabel("Recovery")
    ax.yaxis.grid(True, alpha=0.7)
    ax.set_axisbelow(True)
    save(fig, str(out / "fig5_headroom"))


def main():
    use_paper_style()
    d = json.loads(DATA.read_text(encoding="utf-8"))
    out = HERE
    fig_composition(d, out)
    fig_union(d, out)
    fig_budget(d, out)
    fig_headroom(d, out)
    print("wrote fig2_composition, fig3_union, fig4_budget, fig5_headroom "
          "(.pdf and .png) to", out)


if __name__ == "__main__":
    main()
