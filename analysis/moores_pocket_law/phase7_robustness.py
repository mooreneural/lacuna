"""Phase 7: does the burden effect survive perturbing the analysis?

Six checks. The one the repository cannot support is a criterion sweep over the
"size-robust" rule, because no per-candidate centroid distance is stored, so the
threshold is varied within the Jaccard rule instead and that limit is stated
rather than papered over.

    python analysis/moores_pocket_law/phase7_robustness.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio
from dataio import KS, METHODS

HERE = Path(__file__).resolve().parent
SEED = 0


def spearman_burden_vs_hit(pairs):
    """pairs: [(N, hit)] among covered targets."""
    if len(pairs) < 30:
        return None
    n = np.array([p[0] for p in pairs], float)
    y = np.array([p[1] for p in pairs], float)
    if len(np.unique(y)) < 2:
        return None
    rho, p = stats.spearmanr(n, y)
    return {"rho": float(rho), "p": float(p), "n": len(pairs)}


def main() -> None:
    out = {"seed": SEED}
    cands = dataio.load_candidates()
    folds = dataio.fold_map()
    paired = {s: d for s, d in cands.items() if all(m in d for m in METHODS)}

    print("=" * 76)
    print("R1  Jaccard threshold  (the paper uses 0.25)")
    print("=" * 76)
    r1 = {}
    for thr in (0.15, 0.20, 0.25, 0.30, 0.40):
        pairs = []
        for s, per in paired.items():
            for m in METHODS:
                jl = per[m]
                if any(j >= thr for j in jl):
                    pairs.append((len(jl), int(any(j >= thr for j in jl[:5]))))
        r1["%.2f" % thr] = spearman_burden_vs_hit(pairs)
        print("  thr=%.2f  rho(N, hit@5 | covered) = %+.3f   (n covered = %d)"
              % (thr, r1["%.2f" % thr]["rho"], r1["%.2f" % thr]["n"]))
    out["R1_threshold"] = r1

    print("\n" + "=" * 76)
    print("R2  budget k")
    print("=" * 76)
    r2 = {}
    for k in KS:
        pairs = []
        for s, per in paired.items():
            for m in METHODS:
                jl = per[m]
                if any(j >= dataio.JACCARD_THRESHOLD for j in jl):
                    pairs.append((len(jl),
                                  int(any(j >= dataio.JACCARD_THRESHOLD
                                          for j in jl[:k]))))
        r2[str(k)] = spearman_burden_vs_hit(pairs)
        print("  k=%-3d rho = %+.3f" % (k, r2[str(k)]["rho"]))
    out["R2_budget"] = r2

    print("\n" + "=" * 76)
    print("R3  trimming extreme candidate counts")
    print("=" * 76)
    all_n = sorted(len(per[m]) for per in paired.values() for m in METHODS)
    r3 = {}
    for pct in (0, 1, 5, 10):
        hi = all_n[int((1 - pct / 100) * (len(all_n) - 1))]
        pairs = []
        for s, per in paired.items():
            for m in METHODS:
                jl = per[m]
                if len(jl) > hi:
                    continue
                if any(j >= dataio.JACCARD_THRESHOLD for j in jl):
                    pairs.append((len(jl), int(any(j >= dataio.JACCARD_THRESHOLD
                                                   for j in jl[:5]))))
        r3[str(pct)] = spearman_burden_vs_hit(pairs)
        print("  drop top %2d%% (N > %2d): rho = %+.3f  (n = %d)"
              % (pct, hi, r3[str(pct)]["rho"], r3[str(pct)]["n"]))
    out["R3_trim"] = r3

    print("\n" + "=" * 76)
    print("R4  each detector alone")
    print("=" * 76)
    r4 = {}
    for m in METHODS:
        pairs = [(len(per[m]), int(any(j >= dataio.JACCARD_THRESHOLD
                                       for j in per[m][:5])))
                 for per in paired.values()
                 if any(j >= dataio.JACCARD_THRESHOLD for j in per[m])]
        r4[m] = spearman_burden_vs_hit(pairs)
        print("  %-11s rho = %+.3f  (n = %d)" % (m, r4[m]["rho"], r4[m]["n"]))
    out["R4_per_method"] = r4

    print("\n" + "=" * 76)
    print("R5  burden normalised by protein length (candidate density)")
    print("=" * 76)
    lengths = dataio.protein_lengths()
    dens, raw = [], []
    for s, per in paired.items():
        if s not in lengths:
            continue
        for m in METHODS:
            jl = per[m]
            if not any(j >= dataio.JACCARD_THRESHOLD for j in jl):
                continue
            y = int(any(j >= dataio.JACCARD_THRESHOLD for j in jl[:5]))
            dens.append((len(jl) / lengths[s], y))
            raw.append((len(jl), y))
    rd = spearman_burden_vs_hit(dens)
    rr = spearman_burden_vs_hit(raw)
    print("  raw count      rho = %+.3f" % rr["rho"])
    print("  count / length rho = %+.3f" % rd["rho"])
    out["R5_density"] = {"raw": rr, "density": rd}

    print("\n" + "=" * 76)
    print("R6  train fold versus designated test fold")
    print("=" * 76)
    r6 = {}
    for split in ("train", "test"):
        pairs = []
        for s, per in paired.items():
            if dataio.split_of(s, folds) != split:
                continue
            for m in METHODS:
                jl = per[m]
                if any(j >= dataio.JACCARD_THRESHOLD for j in jl):
                    pairs.append((len(jl), int(any(j >= dataio.JACCARD_THRESHOLD
                                                   for j in jl[:5]))))
        r6[split] = spearman_burden_vs_hit(pairs)
        print("  %-6s rho = %+.3f  (n covered = %d)"
              % (split, r6[split]["rho"], r6[split]["n"]))
    out["R6_split"] = r6

    (HERE / "phase7_robustness.json").write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % (HERE / "phase7_robustness.json"))


if __name__ == "__main__":
    main()
