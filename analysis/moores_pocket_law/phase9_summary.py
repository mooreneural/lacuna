"""Phase 9: collapse every phase output into one summary.json.

    python analysis/moores_pocket_law/phase9_summary.py
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name):
    p = HERE / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    inputs = load("INPUTS.json")
    p2 = load("phase2_reproduction.json")
    p4 = load("phase4_results.json")
    p5 = load("phase5_confounders.json")
    p6 = load("phase6_sweep.json")
    p7 = load("phase7_robustness.json")

    eff = p5["efficiency"]
    etas = [eff[k]["eta"] for k in eff]

    summary = {
        "analysis": "Moore's Pocket Law, falsification-oriented",
        "git_commit": inputs["git_commit"],
        "criterion": "Jaccard >= 0.25, no centroid clause "
                     "(paper/make_data.py:_hit); the repository's wider "
                     "size-robust rule cannot be evaluated because no "
                     "per-candidate centroid distance is stored",
        "cohorts": {
            "paired_test_fold": p2["n"],
            "paired_train_fold": 734,
            "sweep_common_train_targets": p6["n_common_targets"],
        },
        "phase2_reproduction": {
            "status": "exact match to paper/data/analysis.json",
            "methods": {m: {"coverage": v["coverage"], "top5": v["top5"]}
                        for m, v in p2["methods"].items()},
            "union_coverage": p2["union_coverage"],
        },
        "core_finding": {
            "claim": "Conversion of available coverage into top-k recovery "
                     "falls as candidate burden rises.",
            "pooled_spearman_hit5": p4["H1"]["pooled"]["spearman"]["hit_at_5"],
            "logit_beta_standardized_with_method_FE":
                p5["logit_std"]["candidate_count"],
            "protein_length_beta": p5["logit_std"]["protein_length"],
            "within_method_spearman": p5["within_method_spearman"],
            "causal_within_target": p6["covered_at_both"],
            "supported": True,
        },
        "quantitative_form": {
            "claim": "eta = dR/dC declines monotonically with burden "
                     "(d eta / dN < 0).",
            "eta_by_step": eff,
            "eta_sequence": etas,
            "monotonic": all(a > b for a, b in zip(etas, etas[1:])),
            "supported": False,
        },
        "threshold": {
            "claim": "A burden threshold near N=15 separates efficient from "
                     "inefficient regimes.",
            "descriptive_best": p4["H6"]["descriptive_best"],
            "bootstrap_ci": p4["H6"]["bootstrap_ci"],
            "grid_edge_fraction": p4["H6"]["edge_fraction"],
            "model_comparison": p4["H6"]["model_comparison"],
            "supported": False,
        },
        "robustness": p7,
        "failure_composition_at_5": p4["H5"],
        "verdict": "Qualitative relationship supported and causally "
                   "corroborated; quantitative law form and the ~15 threshold "
                   "not supported.",
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1))
    print("wrote %s" % (HERE / "summary.json"))
    print("  core finding supported     :", summary["core_finding"]["supported"])
    print("  quantitative form supported:",
          summary["quantitative_form"]["supported"])
    print("  threshold supported        :", summary["threshold"]["supported"])


if __name__ == "__main__":
    main()
