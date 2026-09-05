"""Not "which pocket is best", but "is this one real". They are different questions.

Asked how you tell a functional pocket from an incidental groove, the honest
answer today is that Lacuna does not tell you. It ranks. A ranker only has to
be monotonic: it can order candidates perfectly while its scores mean nothing
in absolute terms. Deciding whether a site is worth an assay needs the number
itself to be true, and nothing here has ever checked that.

The distinction is not academic. A model that says 0.8 and is right 40% of the
time ranks fine and is useless for triage, and every reported metric in this
project (top-k recovery, coverage, conversion) is invariant to that failure.

"Functional" is operationalised as the only thing structure data can support: a
ligand is observed bound there in a solved structure. That is what the
annotations encode. Its limitation should be stated rather than hidden, because
it is real: a pocket with no annotation is not proven non-functional, it is
merely unobserved, so measured precision is a lower bound.

Three things are measured, and the third is the one that matters:

  discrimination   AUC and average precision. What ranking already needed.
  calibration      Brier score and expected calibration error, against a
                   reliability curve. Does 0.8 mean 80%.
  transfer         A model calibrated on CryptoBench cryptic sites, applied
                   unchanged to COACH420 general sites and to PocketMiner.
                   Calibration that survives only in-domain is a lookup table
                   for one benchmark, not a functional predictor.

Cross-fitting is leave-one-fold-out over the four homology-separated training
folds. The designated test fold is never read here.

    python analysis/moores_pocket_law/calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import dataio
from dataio import JACCARD_THRESHOLD as T
from lacuna.pockets.clusterer import _RANKER_FEATURES, _RANKER_INTERCEPT, _RANKER_WEIGHTS

FEATS = list(_RANKER_FEATURES)
W = np.asarray(_RANKER_WEIGHTS)
B = float(_RANKER_INTERCEPT)
EXT = HERE / "data" / "candidates_ext"
N_BOOT = 20000
N_BINS = 10


def load_train():
    """Train-fold candidates with features, labels and fold, one row per candidate."""
    folds = dataio.fold_map()
    X, y, fold = [], [], []
    for rec in dataio._jsonl(dataio.SWEEP[20]):
        f = folds.get(dataio.pdb_id_of(rec["id"]))
        if f is None or not f.startswith("train"):
            continue
        for c in rec.get("clusters") or []:
            X.append([float(c.get(k, 0.0)) for k in FEATS])
            y.append(1 if float(c.get("jac", 0.0)) >= T else 0)
            fold.append(f)
    return np.asarray(X), np.asarray(y), np.asarray(fold)


def load_external(name: str):
    """Candidates from a dataset the model never trained on."""
    X, y = [], []
    with open(EXT / ("candidates_%s.jsonl" % name), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            fb, jb = r.get("features_by_rank"), r.get("jac_by_rank")
            if not fb or not jb:
                continue
            for c, j in zip(fb, jb):
                X.append([float(c.get(k, 0.0)) for k in FEATS])
                y.append(1 if float(j) >= T else 0)
    return np.asarray(X), np.asarray(y)


# ────────────────────────────── calibration maths ────────────────────────────

def ece(p, y, n_bins=N_BINS):
    """Expected calibration error: mean |confidence - accuracy| over bins.

    Weighted by bin population, so a confident region with few points cannot
    dominate. Equal-width bins, because equal-frequency bins hide exactly the
    sparse high-confidence region that triage depends on.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability(p, y, n_bins=N_BINS):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 5:
            continue
        rows.append((0.5 * (edges[b] + edges[b + 1]), float(p[m].mean()),
                     float(y[m].mean()), int(m.sum())))
    return rows


def report(name, p, y):
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    return {"name": name, "n": int(len(y)), "base_rate": float(y.mean()),
            "auc": float(roc_auc_score(y, p)),
            "ap": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "ece": ece(p, y)}


def line(r):
    return ("%-38s n=%6d  base %.3f  AUC %.3f  AP %.3f  Brier %.4f  ECE %.3f"
            % (r["name"], r["n"], r["base_rate"], r["auc"], r["ap"],
               r["brier"], r["ece"]))


def main() -> None:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X, y, fold = load_train()
    print("train-fold candidates: %d   qualifying: %d (%.1f%%)\n"
          % (len(y), y.sum(), 100 * y.mean()))

    # 1. The shipped ranker's score, squashed. This is what a user would get if
    #    they read the score as a probability, which nothing stops them doing.
    p_ship = 1.0 / (1.0 + np.exp(-(X @ W + B)))

    # 2. Cross-fitted logistic: honestly fitted, but nothing forces calibration.
    p_cv = np.zeros(len(y))
    for held in sorted(set(fold)):
        tr, te = fold != held, fold == held
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
        m.fit(X[tr], y[tr])
        p_cv[te] = m.predict_proba(X[te])[:, 1]

    # 3. Isotonic recalibration, itself cross-fitted so the calibrator never
    #    sees the fold it corrects. Calibrating on your own predictions is the
    #    standard way to report a calibration that does not exist.
    p_iso = np.zeros(len(y))
    for held in sorted(set(fold)):
        tr, te = fold != held, fold == held
        iso = IsotonicRegression(out_of_bounds="clip").fit(p_cv[tr], y[tr])
        p_iso[te] = iso.transform(p_cv[te])

    rows = [report("shipped ranker score (sigmoid)", p_ship, y),
            report("cross-fitted logistic", p_cv, y),
            report("cross-fitted + isotonic", p_iso, y)]
    print("in domain: CryptoBench training folds")
    print("-" * 104)
    for r in rows:
        print(line(r))

    # The shipped score collapses into the lowest bin, so a reliability curve
    # over it has one row and says nothing useful. Report the collapse as a
    # number, and draw the curve for the calibrated model instead, which is the
    # one a caller could actually act on.
    print("")
    print("shipped score read as a probability: mean %.4f, base rate %.4f"
          % (p_ship.mean(), y.mean()))
    print("  %.1f%% of candidates score below 0.01, so every one lands in the "
          "lowest bin" % (100 * (p_ship < 0.01).mean()))
    print("  it orders candidates well (AUC %.3f) and is not a probability"
          % rows[0]["auc"])
    print("")
    print("reliability of the calibrated model, in domain")
    print("  %-14s %-12s %-12s %s" % ("bin", "mean pred", "observed", "n"))
    for _c, mp, obs, cnt in reliability(p_iso, y):
        gap = mp - obs
        flag = "  <-- off by %+.2f" % gap if abs(gap) > 0.10 else ""
        print("  %-14s %-12.3f %-12.3f %d%s"
              % ("%.2f-%.2f" % (max(mp - 0.05, 0), mp + 0.05), mp, obs, cnt, flag))

    # 4. Transfer. One model, fitted and calibrated on all four training folds,
    #    applied unchanged to datasets it has never seen.
    m_full = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
    m_full.fit(X, y)
    iso_full = IsotonicRegression(out_of_bounds="clip").fit(m_full.predict_proba(X)[:, 1], y)

    print("\ntransfer: calibrated on CryptoBench cryptic sites, applied unchanged")
    print("-" * 104)
    ext_rows = []
    for name in ("coach420", "pocketminer"):
        path = EXT / ("candidates_%s.jsonl" % name)
        if not path.exists():
            print("  (%s not present)" % name)
            continue
        Xe, ye = load_external(name)
        if len(ye) < 50 or ye.sum() == 0:
            print("  (%s has too few labelled candidates)" % name)
            continue
        pe = iso_full.transform(m_full.predict_proba(Xe)[:, 1])
        r = report("%s, transferred" % name, pe, ye)
        ext_rows.append(r)
        print(line(r))

    print("\n%s" % ("=" * 104))
    d_ece = rows[0]["ece"] - rows[2]["ece"]
    print("Recalibration moves ECE by %.3f in domain (%.3f -> %.3f)."
          % (d_ece, rows[0]["ece"], rows[2]["ece"]))
    if ext_rows:
        worst = max(ext_rows, key=lambda r: r["ece"])
        if worst["ece"] <= 0.10:
            print("Calibration survives transfer (worst ECE %.3f on %s), so the"
                  % (worst["ece"], worst["name"].split(",")[0]))
            print("score can be read as a probability on data it never saw. That is")
            print("what triage needs and it is a stronger claim than any ranking metric.")
        else:
            print("Calibration does NOT survive transfer (ECE %.3f on %s). The score"
                  % (worst["ece"], worst["name"].split(",")[0]))
            print("is usable for ordering candidates within a protein and not for")
            print("deciding whether a pocket is worth pursuing, which is the question")
            print("that was actually asked.")

    (HERE / "calibration.json").write_text(json.dumps(
        {"in_domain": rows, "transfer": ext_rows,
         "reliability_shipped": reliability(p_ship, y)}, indent=1))
    print("\nwrote %s" % (HERE / "calibration.json"))


if __name__ == "__main__":
    main()
