"""Does 40 conformers beat 20 on data the ranker never saw?

On the train folds, doubling the conformer count looked like a clear win for
Lacuna-PLM: coverage +3.2 [+0.9, +5.5] and recovery +3.2 [+0.8, +5.6], with
conversion holding at 86.3% to 86.9%. The apparent mechanism was that candidate
burden costs a weak ranker and not a strong one, which would have meant better
ranking makes more sampling worth buying.

This is the single confirmatory pass on the designated test fold. Both conformer
counts were collected in one go rather than iterated, because the protocol
allows one look and no more.

The result withdraws the interesting half of the train-fold finding. Coverage
replicates. Recovery does not, for either ranker, and the PLM ranker's
conversion falls on held-out data where it had held on train. That ranker was
fitted on the train candidates, so it ordered them well however many arrived;
unseen, the extra candidates cost it too.

Kept as a script rather than a console session so the numbers behind the
decision not to change the shipped default can be regenerated.

    python analysis/moores_pocket_law/conformer_confirmation.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
T = 0.25
KS = 5
SEED = 0

SETS = {
    "train": (DATA / "candidates_v3" / "candidates_train.jsonl",
              DATA / "conf40" / "candidates_train.jsonl"),
    "test": (DATA / "test_conf20" / "candidates_test.jsonl",
             DATA / "test_conf40" / "candidates_test.jsonl"),
}


def load(path: Path, tool: str):
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r["tool"] == tool:
                out[r["id"]] = [float(x) for x in r["jac_by_rank"]]
    return out


def main() -> None:
    rng = np.random.default_rng(SEED)
    report = {}
    for split, (p20, p40) in SETS.items():
        if not (p20.exists() and p40.exists()):
            print("skipping %s: collection missing" % split)
            continue
        print("\n%s fold" % split.upper())
        report[split] = {}
        for tool in ("lacuna", "lacuna_plm"):
            a, b = load(p20, tool), load(p40, tool)
            ids = sorted(set(a) & set(b))
            if not ids:
                continue

            def stats(d):
                cov = np.array([any(x >= T for x in d[i]) for i in ids])
                hit = np.array([int(any(x >= T for x in d[i][:KS])) for i in ids])
                return cov, hit, float(np.mean([len(d[i]) for i in ids]))

            ca, ha, na = stats(a)
            cb, hb, nb = stats(b)
            bi = rng.integers(0, len(ids), size=(20000, len(ids)))

            print("  %s  (paired on %d targets)" % (tool, len(ids)))
            print("    %-5s %6s %9s %8s %10s"
                  % ("conf", "meanN", "coverage", "conv@5", "recov@5"))
            cells = {}
            for lab, c, h, n in (("20", ca, ha, na), ("40", cb, hb, nb)):
                conv = float(h[c].mean()) if c.any() else float("nan")
                cells[lab] = {"mean_N": n, "coverage": float(c.mean()),
                              "conv_5": conv, "recovery_5": float(h.mean())}
                print("    %-5s %6.1f %8.1f%% %7.1f%% %9.1f%%"
                      % (lab, n, 100 * c.mean(), 100 * conv, 100 * h.mean()))

            deltas = {}
            for lab, x, y in (("coverage", cb.astype(int), ca.astype(int)),
                              ("recovery_5", hb, ha)):
                d = x - y
                bm = np.sort(d[bi].mean(axis=1))
                lo, hi = float(bm[500]), float(bm[19499])
                deltas[lab] = {"delta": float(d.mean()), "ci": [lo, hi]}
                star = "  EXCLUDES 0" if (lo > 0 or hi < 0) else ""
                print("      delta %-11s %+6.1f  [%+5.1f, %+5.1f]%s"
                      % (lab, 100 * d.mean(), 100 * lo, 100 * hi, star))
            report[split][tool] = {"n": len(ids), "levels": cells, "deltas": deltas}

    (HERE / "conformer_confirmation.json").write_text(json.dumps(report, indent=1))
    print("\nwrote %s" % (HERE / "conformer_confirmation.json"))
    print("\nDecision: leave the shipped default at 20 conformers. Coverage")
    print("replicates on held-out data; recovery does not, and runtime doubles.")


if __name__ == "__main__":
    main()
