"""Reproduce the paper's +27.2 point gain and its 95% CI with the repo's own method."""
import io, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # benchmarks/provenance/ -> repo root
sys.path.insert(0, str(ROOT / "benchmarks"))
from train_ranker import _paired_ci

def jl(p, tool):
    out = {}
    for line in io.open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r["tool"] == tool:
            out[r["id"]] = r
    return out

R = ROOT / "benchmarks"
L = jl(R / "detectors_cb_current.jsonl", "lacuna_learned")
C = jl(R / "provenance" / "detectors_cb_r15crypticity_180.jsonl", "lacuna_nma")
ids = sorted(set(L) & set(C))
print("paired ids", len(ids))
hl = [L[i]["jaccard"] >= 0.25 for i in ids]
hc = [C[i]["jaccard"] >= 0.25 for i in ids]
print("learned    %d/%d = %.1f%%" % (sum(hl), len(hl), 100*sum(hl)/len(hl)))
print("crypticity %d/%d = %.1f%%" % (sum(hc), len(hc), 100*sum(hc)/len(hc)))
d, lo, hi = _paired_ci(hl, hc)
print("paired diff %+.1f  CI [%+.1f, %+.1f]  (n_boot=10000, seed=0)" % (100*d, 100*lo, 100*hi))
