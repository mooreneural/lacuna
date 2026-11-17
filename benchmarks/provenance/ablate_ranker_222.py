"""Full-fold ranking ablation: analytic crypticity against the learned ranker."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # benchmarks/provenance/ -> repo root
sys.path.insert(0, str(ROOT / "benchmarks"))
from train_ranker import _paired_ci

LEARNED = ROOT / "benchmarks" / "cryptobench_results.json"
CRYPTIC = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    ROOT / "benchmarks" / "cryptobench_results_crypticity.json")

def load(p):
    d = json.loads(p.read_text(encoding="utf-8"))
    return d["summary"], {r["id"] + r["chain"]: r for r in d["structures"]}, d["unscored"]

sl, L, ul = load(LEARNED)
sc, C, uc = load(CRYPTIC)
print("learned   ", {k: sl[k] for k in ("conformers","rank_by","backend","max_residues","n_scored","n_unscored","wall_minutes")})
print("crypticity", {k: sc[k] for k in ("conformers","rank_by","backend","max_residues","n_scored","n_unscored","wall_minutes")})
print("unscored:", len(ul), len(uc))

ids = sorted(set(L) & set(C))
print("\npaired ids: %d  (learned %d, crypticity %d)" % (len(ids), len(L), len(C)))

# Candidate set held fixed: the clustering runs before the sort, so an identical
# candidate set shows up as an identical cluster count and an identical
# order-independent best-over-all-clusters overlap.
diff = []
for i in ids:
    a, b = L[i], C[i]
    for f in ("n_residues", "n_known", "n_clusters", "site_radius_a"):
        if a[f] != b[f]:
            diff.append((i, f, a[f], b[f]))
    for f in ("jaccard", "recall", "centroid_dist_a", "found"):
        if a["any_rank"][f] != b["any_rank"][f]:
            diff.append((i, "any_rank." + f, a["any_rank"][f], b["any_rank"][f]))
print("candidate-set differences (n_clusters, n_residues, n_known, site_radius, any-rank best):", len(diff))
for d in diff[:20]:
    print("   ", d)

def arm(R, key):
    if key == "jaccard":
        return [R[i]["top5"]["jaccard"] >= 0.25 for i in ids]
    return [R[i]["top5"]["found"] for i in ids]

print("\n%-26s %-18s %-18s %s" % ("criterion", "learned", "crypticity", "paired difference"))
res = {}
for key, label in (("jaccard", "Jaccard only >= 0.25"),
                   ("found", "centroid <=4A or Jac")):
    hl, hc = arm(L, key), arm(C, key)
    d, lo, hi = _paired_ci(hl, hc)
    res[key] = (sum(hl), sum(hc), 100*d, 100*lo, 100*hi)
    print("%-26s %4d/%d (%.1f%%)  %4d/%d (%.1f%%)  %+.1f pts  95%% CI [%+.1f, %+.1f]" % (
        label, sum(hl), len(hl), 100*sum(hl)/len(hl),
        sum(hc), len(hc), 100*sum(hc)/len(hc), 100*d, 100*lo, 100*hi))
    print("%-26s learned-only %d  crypticity-only %d  both %d  neither %d" % (
        "", sum(1 for x,y in zip(hl,hc) if x and not y),
        sum(1 for x,y in zip(hl,hc) if y and not x),
        sum(1 for x,y in zip(hl,hc) if x and y),
        sum(1 for x,y in zip(hl,hc) if not x and not y)))
    print("%-26s ratio learned/crypticity %.3f" % ("", sum(hl)/max(sum(hc),1)))

print("\nother columns, crypticity arm (222):")
print("  top5 legacy recall>=0.30 : %d/%d (%.1f%%)" % (sc["top5_found_legacy"], sc["n_scored"], 100*sc["top5_found_legacy"]/sc["n_scored"]))
print("  top8 cen-or-jac          : %d/%d (%.1f%%)" % (sc["top8_found"], sc["n_scored"], 100*sc["top8_found"]/sc["n_scored"]))
print("  coverage ceiling any rank: %d/%d (%.1f%%)" % (sc["coverage_ceiling_found"], sc["n_scored"], 100*sc["coverage_ceiling_found"]/sc["n_scored"]))
print("  top8 jaccard only        : %d/%d" % (sum(C[i]["top8"]["jaccard"] >= 0.25 for i in ids), len(ids)))
print("  ceiling jaccard only     : %d/%d" % (sum(C[i]["any_rank"]["jaccard"] >= 0.25 for i in ids), len(ids)))
print("learned arm (222) for reference:")
print("  top5 legacy              : %d/%d" % (sl["top5_found_legacy"], sl["n_scored"]))
print("  top8 jaccard only        : %d/%d" % (sum(L[i]["top8"]["jaccard"] >= 0.25 for i in ids), len(ids)))
print("  ceiling jaccard only     : %d/%d" % (sum(L[i]["any_rank"]["jaccard"] >= 0.25 for i in ids), len(ids)))

# the old 180 sub-cohort, for the cohort-to-cohort comparison
old = [i for i in ids if not L[i]["composite"] and L[i]["n_residues"] <= 700]
print("\nsub-cohort that the old harness scored (%d):" % len(old))
for key, label in (("jaccard", "Jaccard only"), ("found", "cen-or-jac")):
    hl = [L[i]["top5"]["jaccard"] >= 0.25 if key == "jaccard" else L[i]["top5"]["found"] for i in old]
    hc = [C[i]["top5"]["jaccard"] >= 0.25 if key == "jaccard" else C[i]["top5"]["found"] for i in old]
    d, lo, hi = _paired_ci(hl, hc)
    print("  %-14s learned %d/%d  crypticity %d/%d  %+.1f pts CI [%+.1f, %+.1f]" % (
        label, sum(hl), len(hl), sum(hc), len(hc), 100*d, 100*lo, 100*hi))
