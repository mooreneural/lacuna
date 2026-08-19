"""Phase 3: build the tidy target x method analysis table.

    python analysis/moores_pocket_law/phase3_table.py
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dataio

HERE = Path(__file__).resolve().parent
FIELDS = ["target_id", "pdb", "chain", "method", "split", "protein_length",
          "site_size", "total_candidate_count", "qualifying_candidate_exists",
          "best_qualifying_rank", "number_of_qualifying_candidates",
          "best_jaccard", "best_centroid_distance", "candidate_density",
          "hit_at_1", "hit_at_3", "hit_at_5", "hit_at_10", "hit_at_20"]


def main() -> None:
    rows = dataio.build_table()
    out = HERE / "results.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_targets = len({r["target_id"] for r in rows})
    print("wrote %s" % out)
    print("  rows: %d  (%d targets x %d methods)"
          % (len(rows), n_targets, len(dataio.METHODS)))
    print("  split: %s" % dict(Counter(r["split"] for r in rows)))

    missing_len = sum(1 for r in rows if r["protein_length"] is None)
    missing_site = sum(1 for r in rows if r["site_size"] is None)
    print("  missing protein_length: %d rows (%.1f%%)"
          % (missing_len, 100 * missing_len / len(rows)))
    print("  missing site_size     : %d rows" % missing_site)
    print("  best_centroid_distance: null for all rows, not stored upstream")

    print("\ncandidate burden by method (all splits):")
    for m in dataio.METHODS:
        n = sorted(r["total_candidate_count"] for r in rows if r["method"] == m)
        print("  %-11s min=%2d  p25=%3d  median=%3d  p75=%3d  max=%3d"
              % (m, n[0], n[len(n) // 4], n[len(n) // 2],
                 n[3 * len(n) // 4], n[-1]))


if __name__ == "__main__":
    main()
