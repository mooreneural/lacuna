"""Measure Lacuna wall-clock against chain length, for the scaling figure.

Single chains, extracted to a temp PDB before timing. Lacuna processes whatever
is in the file, so timing a multi-chain entry while plotting against its longest
chain would understate the work and make the curve meaningless. Extracting one
chain also matches how the CryptoBench benchmark runs the tool.

Real timings on one machine, one core, NMA backend at the default 20 conformers,
each in a fresh subprocess so interpreter warm-up cannot flatter later runs.
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
import time
from pathlib import Path

import gemmi

REPO = Path(r"C:\Users\clayt\Documents\GitHub\lacuna")
CIF = REPO / "benchmarks/cb_data/cif"
OUT = Path(__file__).parent
CHAINS = OUT / "chains"
N_TARGETS = 30
CONFORMERS = 20
MAX_RES = 700


def extract_longest_chain(path: Path, dest: Path) -> int:
    """Write the longest amino-acid chain to `dest`; return its length."""
    st = gemmi.read_structure(str(path))
    st.setup_entities()
    st.remove_ligands_and_waters()
    st.remove_alternative_conformations()
    st.remove_hydrogens()
    best, best_n = None, 0
    for ch in st[0]:
        n = sum(1 for r in ch
                if (i := gemmi.find_tabulated_residue(r.name)) and i.is_amino_acid())
        if n > best_n:
            best, best_n = ch.name, n
    if best is None or best_n < 50:
        return 0
    for model in st:
        for name in [c.name for c in model]:
            if name != best:
                model.remove_chain(name)
    st.write_pdb(str(dest))
    return best_n


def main() -> None:
    CHAINS.mkdir(exist_ok=True)
    files = sorted(CIF.glob("*.cif"))
    random.Random(0).shuffle(files)

    picked: list[tuple[Path, int]] = []
    buckets: dict[int, int] = {}
    for f in files:
        if len(picked) >= N_TARGETS:
            break
        dest = CHAINS / f"{f.stem}_A.pdb"
        try:
            n = extract_longest_chain(f, dest)
        except Exception:
            continue
        if not (50 <= n <= MAX_RES):
            continue
        b = n // 100
        if buckets.get(b, 0) >= 5:          # spread across the size range
            continue
        buckets[b] = buckets.get(b, 0) + 1
        picked.append((dest, n))

    picked.sort(key=lambda p: p[1])
    print(f"timing {len(picked)} single chains, {CONFORMERS} conformers, NMA",
          flush=True)

    rows = []
    for i, (f, n) in enumerate(picked, 1):
        outdir = OUT / "sweep_tmp" / f.stem
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "lacuna.cli", "discover", str(f),
             "--backend", "nma", "--conformers", str(CONFORMERS),
             "--output", str(outdir), "--quiet"],
            cwd=REPO, capture_output=True, text=True, timeout=1200)
        dt = time.perf_counter() - t0
        ok = proc.returncode == 0
        n_pockets = 0
        rep = outdir / "pocket_report.json"
        if ok and rep.exists():
            try:
                n_pockets = json.loads(rep.read_text())["n_pockets_found"]
            except Exception:
                pass
        rows.append({"id": f.stem, "n_res": n, "seconds": round(dt, 2),
                     "ok": ok, "n_pockets": n_pockets,
                     "err": "" if ok else proc.stderr[-200:]})
        print(f"[{i}/{len(picked)}] {f.stem} n={n} {dt:.1f}s ok={ok}", flush=True)
        (OUT / "runtime.json").write_text(json.dumps(rows, indent=1))

    good = [r for r in rows if r["ok"]]
    print(f"\n{len(good)}/{len(rows)} succeeded")
    if good:
        ts = sorted(r["seconds"] for r in good)
        print(f"seconds: min {ts[0]:.1f}  median {ts[len(ts)//2]:.1f}  max {ts[-1]:.1f}")


if __name__ == "__main__":
    main()
