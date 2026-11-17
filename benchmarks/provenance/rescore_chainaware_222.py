"""Chain-aware rescoring of the 222-structure full fold.

Produces every figure in Section 11 of
benchmarks/PROVENANCE_cryptobench_full_fold_2026-11-16.md, including the
paired bootstrap, and writes them to section11_numbers.json beside this
file. Run it from the repository root:

    python -I benchmarks/provenance/rescore_chainaware_222.py
"""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BENCH = (ROOT / "benchmarks").as_posix() + "/"
SCR = Path(__file__).resolve().parent.as_posix() + "/"

spec = importlib.util.spec_from_file_location("tr", BENCH + "train_ranker.py")
tr = importlib.util.module_from_spec(spec); spec.loader.exec_module(tr)

L = {r["id"] + str(r["chain"]): r for r in
     json.load(open(BENCH + "cryptobench_results_chainaware.json"))["structures"]}
C = {r["id"] + str(r["chain"]): r for r in
     json.load(open(BENCH + "cryptobench_results_crypticity_chainaware.json"))["structures"]}
assert set(L) == set(C)
ids = sorted(L)
JT, CT = 0.25, 4.0

def dist(r, sec):
    d = r[sec]["centroid_dist_a"]
    return float("inf") if d is None else d
def hit(r, sec, mode, kind):
    j = r[sec]["jaccard"] if mode == "aware" else r[sec]["jaccard_chain_blind"]
    return int(j >= JT if kind == "jac" else (dist(r, sec) <= CT or j >= JT))

COH = {
 "full222":   ids,
 "iface38":   [k for k in ids if L[k]["composite"]],
 "single184": [k for k in ids if not L[k]["composite"]],
 "old180":    [k for k in ids if not L[k]["composite"] and L[k]["n_residues"] <= 700],
 "new42":     [k for k in ids if L[k]["composite"] or L[k]["n_residues"] > 700],
 "over700_4": [k for k in ids if not L[k]["composite"] and L[k]["n_residues"] > 700],
}

res = {}
for cn, ks in COH.items():
    n = len(ks)
    for mode in ("blind", "aware"):
        for kind in ("jac", "cenjac"):
            t5 = sum(hit(L[k], "top5", mode, kind) for k in ks)
            t8 = sum(hit(L[k], "top8", mode, kind) for k in ks)
            cv = sum(hit(L[k], "any_rank", mode, kind) for k in ks)
            c5 = sum(hit(C[k], "top5", mode, kind) for k in ks)
            c8 = sum(hit(C[k], "top8", mode, kind) for k in ks)
            ccv = sum(hit(C[k], "any_rank", mode, kind) for k in ks)
            a = [hit(L[k], "top5", mode, kind) for k in ks]
            b = [hit(C[k], "top5", mode, kind) for k in ks]
            d, lo, hi = tr._paired_ci(a, b, n_boot=10000, seed=0)
            res[(cn, mode, kind)] = dict(
                n=n, t5=t5, t8=t8, cv=cv, none=n-cv, r68=t8-t5,
                p5=100*t5/n, p8=100*t8/n, pcv=100*cv/n, pnone=100*(n-cv)/n,
                c5=c5, c8=c8, ccv=ccv, cp5=100*c5/n,
                d=100*d, lo=100*lo, hi=100*hi,
                L_only=sum(1 for x,y in zip(a,b) if x and not y),
                C_only=sum(1 for x,y in zip(a,b) if y and not x),
                both=sum(1 for x,y in zip(a,b) if x and y),
                neither=sum(1 for x,y in zip(a,b) if not x and not y),
                ratio=(t5/c5) if c5 else None,
                legacy5=sum(1 for k in ks if (L[k]["top5"]["recall"] if mode=="aware"
                            else L[k]["top5"]["recall_chain_blind"]) >= 0.30),
                c_legacy5=sum(1 for k in ks if (C[k]["top5"]["recall"] if mode=="aware"
                            else C[k]["top5"]["recall_chain_blind"]) >= 0.30),
            )

# candidate-set invariance between arms
cand_diff = 0
for k in ids:
    a, b = L[k], C[k]
    if (a["n_clusters"], a["n_residues"], a["n_known"], a["n_known_chain_blind"]) != \
       (b["n_clusters"], b["n_residues"], b["n_known"], b["n_known_chain_blind"]):
        cand_diff += 1
    if a["any_rank"]["jaccard"] != b["any_rank"]["jaccard"] or \
       a["any_rank"]["jaccard_chain_blind"] != b["any_rank"]["jaccard_chain_blind"] or \
       a["any_rank"]["centroid_dist_a"] != b["any_rank"]["centroid_dist_a"]:
        cand_diff += 1

# reproducibility of the learned arm against the stored chain-blind run
OLD = {r["id"] + str(r["chain"]): r for r in
       json.load(open(BENCH + "cryptobench_results.json"))["structures"]}
repro = 0
for k in ids:
    for sec in ("top5", "top8", "any_rank"):
        if (OLD[k][sec]["jaccard"] != L[k][sec]["jaccard_chain_blind"]
            or OLD[k][sec]["recall"] != L[k][sec]["recall_chain_blind"]
            or OLD[k][sec]["centroid_dist_a"] != L[k][sec]["centroid_dist_a"]):
            repro += 1

# invariance on single-chain cohorts
inv = {}
for cn in ("single184", "old180"):
    bad = 0
    for k in COH[cn]:
        for sec in ("top5", "top8", "any_rank"):
            for r in (L[k], C[k]):
                if r[sec]["jaccard"] != r[sec]["jaccard_chain_blind"] or \
                   r[sec]["recall"] != r[sec]["recall_chain_blind"]:
                    bad += 1
    inv[cn] = bad

# per-structure losses, learned arm, full fold
losses = []
for k in ids:
    if hit(L[k], "top5", "blind", "jac") and not hit(L[k], "top5", "aware", "jac"):
        r = L[k]
        losses.append((r["id"], r["chain"], r["n_known"], r["n_known_chain_blind"],
                       r["top5"]["jaccard_chain_blind"], r["top5"]["jaccard"],
                       r["top5"]["centroid_dist_a"]))
moved = sum(1 for k in COH["iface38"]
            if L[k]["top5"]["jaccard"] != L[k]["top5"]["jaccard_chain_blind"])
moved_any = sum(1 for k in COH["iface38"]
                if any(L[k][s]["jaccard"] != L[k][s]["jaccard_chain_blind"]
                       for s in ("top5", "top8", "any_rank")))

json.dump({"res": {"|".join(kk): vv for kk, vv in res.items()},
           "cand_diff": cand_diff, "repro_field_diffs": repro, "invariance": inv,
           "losses": losses, "iface_moved_top5": moved, "iface_moved_any": moved_any},
          open(SCR + "section11_numbers.json", "w"), indent=1)

def P(*a): print(*a)
for key in sorted(res):
    P("|".join(key), json.dumps(res[key]))
P("cand_diff", cand_diff, "repro_field_diffs", repro, "invariance", inv)
P("iface_moved_top5", moved, "iface_moved_any", moved_any)
for l in losses: P("LOSS", l)
