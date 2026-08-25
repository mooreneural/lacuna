"""Phase 12: how much does the answer depend on the machine?

Lacuna's ensemble comes from an elastic-network normal mode analysis, which
eigendecomposes a Hessian. Eigenvector sign and the ordering of near-degenerate
modes are decided by the LAPACK backend, not by the seed, so the same code with
the same seed on the same input can produce a different ensemble on a different
platform. Different conformers mean different pockets.

This measures the size of that effect on the designated test fold by running the
identical commit and seed under Windows and under WSL and pairing on structure.

The point is not that one platform is right. It is that "regenerates from source"
is a claim about a platform as well as a repository, and the paper should say so
with a number attached.

    python analysis/moores_pocket_law/phase12_platform_effect.py \
        --a <windows candidates.jsonl> --b <wsl candidates.jsonl>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataio import JACCARD_THRESHOLD as T

SEED = 0
KS = (1, 3, 5, 10, 20)


def load(path, tool="lacuna"):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if r["tool"] == tool:
                out[r["id"]] = r
    return out


def paired_ci(a, b, n_boot=20000, seed=SEED):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = np.sort(d[idx].mean(axis=1))
    return float(d.mean()), float(m[int(0.025 * n_boot)]), float(m[int(0.975 * n_boot) - 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="platform A dump (Windows)")
    ap.add_argument("--b", required=True, help="platform B dump (WSL)")
    ap.add_argument("--label-a", default="Windows")
    ap.add_argument("--label-b", default="WSL")
    ap.add_argument("--tool", default="lacuna")
    args = ap.parse_args()

    A, B = load(args.a, args.tool), load(args.b, args.tool)
    ids = sorted(set(A) & set(B))
    print("paired on %d structures (%s: %d, %s: %d)\n"
          % (len(ids), args.label_a, len(A), args.label_b, len(B)))

    identical = sum(A[i]["jac_by_rank"] == B[i]["jac_by_rank"] for i in ids)
    same_n = sum(A[i]["n_prop"] == B[i]["n_prop"] for i in ids)
    print("=" * 72)
    print("agreement between platforms")
    print("=" * 72)
    print("  byte-identical candidate lists : %d / %d  (%.1f%%)"
          % (identical, len(ids), 100 * identical / len(ids)))
    print("  identical candidate counts     : %d / %d  (%.1f%%)"
          % (same_n, len(ids), 100 * same_n / len(ids)))

    na = np.array([A[i]["n_prop"] for i in ids], float)
    nb = np.array([B[i]["n_prop"] for i in ids], float)
    m, lo, hi = paired_ci(na, nb)
    print("  candidate count, %s minus %s: %+.2f [%+.2f, %+.2f]"
          % (args.label_a, args.label_b, m, lo, hi))

    print("\n" + "=" * 72)
    print("does it change the reported result?")
    print("=" * 72)
    print("  %-6s %12s %12s %26s" % ("k", args.label_a, args.label_b, "paired delta (95% CI)"))
    out = {"n": len(ids), "identical_lists": identical, "identical_counts": same_n,
           "by_k": {}}
    for k in KS:
        a = [any(j >= T for j in A[i]["jac_by_rank"][:k]) for i in ids]
        b = [any(j >= T for j in B[i]["jac_by_rank"][:k]) for i in ids]
        m, lo, hi = paired_ci(a, b)
        flag = "  <-- excludes 0" if (lo > 0 or hi < 0) else ""
        print("  k=%-4d %11.1f%% %11.1f%%   %+6.1f [%+5.1f, %+5.1f]%s"
              % (k, 100 * np.mean(a), 100 * np.mean(b), 100 * m, 100 * lo, 100 * hi, flag))
        out["by_k"][str(k)] = {"a": float(np.mean(a)), "b": float(np.mean(b)),
                               "delta": m, "ci": [lo, hi]}

    cov_a = [any(j >= T for j in A[i]["jac_by_rank"]) for i in ids]
    cov_b = [any(j >= T for j in B[i]["jac_by_rank"]) for i in ids]
    m, lo, hi = paired_ci(cov_a, cov_b)
    print("  %-6s %11.1f%% %11.1f%%   %+6.1f [%+5.1f, %+5.1f]"
          % ("cover", 100 * np.mean(cov_a), 100 * np.mean(cov_b),
             100 * m, 100 * lo, 100 * hi))
    out["coverage"] = {"a": float(np.mean(cov_a)), "b": float(np.mean(cov_b)),
                       "delta": m, "ci": [lo, hi]}

    print("\n  How to read this: if the intervals span zero, the platforms differ")
    print("  per structure but not in aggregate, which is the reassuring outcome.")
    print("  It still means a reproduction on the other platform will not match")
    print("  structure by structure.")

    dst = Path(__file__).resolve().parent / "phase12_platform.json"
    dst.write_text(json.dumps(out, indent=1))
    print("\nwrote %s" % dst)


if __name__ == "__main__":
    main()
