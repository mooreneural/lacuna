"""Phase 6: the within-target burden manipulation.

Everything in phase4 is observational: candidate burden is a property of a
method-target pair, so a declining conversion curve is equally consistent with
competition and with selection on coverage.

The conformer sweep breaks that. Running Lacuna at 1, 3, 5, 10 and 20 conformers
produces different candidate burdens on the *same* targets with the same
detector, ranker and criterion, so burden is manipulated rather than observed.

One honest caveat: raising the conformer count changes the candidate set as well
as its size, since more conformers sample more conformations. So this is not a
pure burden manipulation either. What it does remove is cross-method and
cross-target confounding, which is what the pooled analysis cannot escape. The
decisive comparison is restricted further still, to targets covered at *both*
ends of the sweep: for those the site was findable at either burden, so a drop
in top-k there cannot be a coverage-selection artifact.

Train fold only, so nothing here touches the designated test fold.

    python analysis/moores_pocket_law/phase6_sweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio
from dataio import JACCARD_THRESHOLD, KS, SWEEP, boot_ci

HERE = Path(__file__).resolve().parent
SEED = 0


def hit(jacs, k):
    return any(j >= JACCARD_THRESHOLD for j in jacs[:k])


def covered(jacs):
    return any(j >= JACCARD_THRESHOLD for j in jacs)


def paired_delta_ci(a, b, n_boot=20000, seed=SEED):
    """Bootstrap CI on the paired mean difference a - b, resampling targets."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = np.sort(d[idx].mean(axis=1))
    return (float(d.mean()), float(m[int(0.025 * n_boot)]),
            float(m[int(0.975 * n_boot) - 1]))


def main() -> None:
    folds = dataio.fold_map()
    levels = sorted(SWEEP)
    ranked = {}
    for n_conf, fname in SWEEP.items():
        d = dataio.lacuna_ranked(dataio.DATA / fname)
        ranked[n_conf] = {s: j for s, j in d.items()
                          if dataio.split_of(s, folds) == "train"}
        print("conf_%-2d loaded %4d train structures" % (n_conf, len(ranked[n_conf])))

    common = sorted(set.intersection(*(set(ranked[n]) for n in levels)))
    print("\ncommon train-fold targets across the whole sweep: %d\n" % len(common))

    out = {"seed": SEED, "n_common_targets": len(common), "levels": {},
           "paired_vs_20": {}, "covered_at_both": {}}

    print("=" * 78)
    print("burden, coverage and conversion as the ensemble grows")
    print("=" * 78)
    print("%-6s %8s %10s %10s %12s" %
          ("conf", "mean N", "coverage", "top-5", "conversion@5"))
    for n_conf in levels:
        jl = ranked[n_conf]
        N = float(np.mean([len(jl[s]) for s in common]))
        cov = float(np.mean([covered(jl[s]) for s in common]))
        t5 = float(np.mean([hit(jl[s], 5) for s in common]))
        conv = t5 / cov if cov else float("nan")
        out["levels"][str(n_conf)] = {
            "mean_N": N, "coverage": cov, "top5": t5, "conversion_at_5": conv,
            "topk": {str(k): float(np.mean([hit(jl[s], k) for s in common]))
                     for k in KS}}
        print("%-6d %8.1f %9.1f%% %9.1f%% %11.1f%%"
              % (n_conf, N, 100 * cov, 100 * t5, 100 * conv))

    # --- the decisive comparison -------------------------------------------
    print("\n" + "=" * 78)
    print("targets covered at BOTH ends of the sweep: does the larger")
    print("candidate set rank the same findable site worse?")
    print("=" * 78)
    lo_lvl, hi_lvl = levels[0], levels[-1]
    both = [s for s in common
            if covered(ranked[lo_lvl][s]) and covered(ranked[hi_lvl][s])]
    print("conf_%d vs conf_%d, both covered: n = %d" % (lo_lvl, hi_lvl, len(both)))
    print("  mean N: %.1f -> %.1f\n"
          % (np.mean([len(ranked[lo_lvl][s]) for s in both]),
             np.mean([len(ranked[hi_lvl][s]) for s in both])))
    print("  %-5s %12s %12s %26s" % ("k", "conf_%d" % lo_lvl,
                                     "conf_%d" % hi_lvl, "paired delta (95% CI)"))
    for k in KS:
        a = [hit(ranked[hi_lvl][s], k) for s in both]
        b = [hit(ranked[lo_lvl][s], k) for s in both]
        d, dlo, dhi = paired_delta_ci(a, b)
        out["covered_at_both"]["hit_at_%d" % k] = {
            "n": len(both), "low": float(np.mean(b)), "high": float(np.mean(a)),
            "delta": d, "delta_ci": [dlo, dhi]}
        star = "  <-- excludes 0" if (dlo > 0 or dhi < 0) else ""
        print("  k=%-3d %11.1f%% %11.1f%%   %+6.1f [%+5.1f, %+5.1f]%s"
              % (k, 100 * np.mean(b), 100 * np.mean(a),
                 100 * d, 100 * dlo, 100 * dhi, star))

    # Rank depth of the same site under both burdens.
    ra, rb = [], []
    for s in both:
        ra.append(dataio.first_qualifying_rank(ranked[hi_lvl][s]))
        rb.append(dataio.first_qualifying_rank(ranked[lo_lvl][s]))
    d, dlo, dhi = paired_delta_ci(ra, rb)
    out["covered_at_both"]["rank_delta"] = {"delta": d, "delta_ci": [dlo, dhi]}
    print("\n  rank of first qualifying candidate, conf_%d minus conf_%d:"
          % (hi_lvl, lo_lvl))
    print("    %+.2f ranks [%+.2f, %+.2f]   (median %d -> %d)"
          % (d, dlo, dhi, int(np.median(rb)), int(np.median(ra))))

    # --- every adjacent step, to see whether the effect is monotone ---------
    print("\n" + "=" * 78)
    print("adjacent steps, paired on targets covered at both levels")
    print("=" * 78)
    print("  %-14s %6s %9s %26s" % ("step", "n", "dN", "d hit@5 (95% CI)"))
    for lo_l, hi_l in zip(levels, levels[1:]):
        sel = [s for s in common
               if covered(ranked[lo_l][s]) and covered(ranked[hi_l][s])]
        a = [hit(ranked[hi_l][s], 5) for s in sel]
        b = [hit(ranked[lo_l][s], 5) for s in sel]
        dN = (np.mean([len(ranked[hi_l][s]) for s in sel])
              - np.mean([len(ranked[lo_l][s]) for s in sel]))
        d, dlo, dhi = paired_delta_ci(a, b)
        out["paired_vs_20"]["%d_to_%d" % (lo_l, hi_l)] = {
            "n": len(sel), "delta_N": float(dN), "delta_hit5": d,
            "ci": [dlo, dhi]}
        star = "  <-- excludes 0" if (dlo > 0 or dhi < 0) else ""
        print("  conf_%-2d -> %-3d %6d %+9.1f   %+6.1f [%+5.1f, %+5.1f]%s"
              % (lo_l, hi_l, len(sel), dN, 100 * d, 100 * dlo, 100 * dhi, star))

    (HERE / "phase6_sweep.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "phase6_sweep.json"))


if __name__ == "__main__":
    main()
