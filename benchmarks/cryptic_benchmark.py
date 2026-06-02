"""Cryptic pocket benchmark — apo/holo PDB pairs.

For each entry we either supply known binding-site residues directly (taken
from the published literature) OR supply a holo PDB ID and let the script
auto-extract the binding site as protein residues within HOLO_CUTOFF Å of
the principal ligand.

Success criterion (same as run_benchmarks.py): ≥30% residue overlap with the
known site in the top-5 Lacuna pocket clusters.

Usage:
    python benchmarks/cryptic_benchmark.py              # full run, 20 conformers
    python benchmarks/cryptic_benchmark.py --quick      # 10 conformers, faster
    python benchmarks/cryptic_benchmark.py --category cryptic
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

CONFORMERS = 20
HOLO_CUTOFF = 4.5          # Å — residue within this of any ligand atom → binding site
OVERLAP_THRESHOLD = 0.30   # success if best overlap in top-5 ≥ this

# HETATM residue names to ignore when selecting the "principal ligand"
SOLVENT_CODES = frozenset({
    "HOH", "WAT", "DOD",                          # water
    "SO4", "SUL", "SF4",                           # sulfate
    "PO4", "HPO", "H2P",                           # phosphate
    "EDO", "EGL",                                   # ethylene glycol
    "GOL", "PGO",                                   # glycerol
    "ACT", "ACE", "ACY", "ACM",                    # acetate / acetyl
    "FMT",                                          # formate
    "CIT", "TLA", "TAR",                            # citrate / tartrate
    "MES", "HEP", "TRS", "BIS", "TRIS",            # buffers
    "DMF", "DMS", "DIO", "IPA",                     # organic solvents
    "CL", "NA", "MG", "ZN", "CA", "K",
    "FE", "MN", "CO", "CU", "NI", "CD", "HG",      # ions
    "PE3", "PE4", "PE5", "PE6", "PE7", "PE8",       # PEG variants
    "PEG", "P33", "P6G",
    "BOG", "BNG",                                   # detergents
    "NI", "CO",
})

# ── dataset ───────────────────────────────────────────────────────────────────

DATASET = [
    # ── CRYPTIC POCKETS ──────────────────────────────────────────────────────
    # Pocket absent or too small to detect on the apo structure alone;
    # only accessible via conformational sampling.

    {
        "id": "T4L_L99A",
        "name": "T4 Lysozyme L99A (hydrophobic cavity)",
        "category": "cryptic",
        "apo_pdb": "1L90", "apo_chain": "A",
        # Known lining residues from Eriksson/Mobley literature
        "known_residues": {99, 102, 106, 111, 118, 121, 133, 153},
        "citation": "Eriksson 1992; Mobley 2007",
    },
    {
        "id": "KRAS_SIIP",
        "name": "K-Ras WT apo (switch-II cryptic pocket)",
        "category": "cryptic",
        "apo_pdb": "4OBE", "apo_chain": "A",
        "known_residues": {12, 13, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
        "citation": "Ostrem 2013",
    },
    {
        "id": "IL2",
        "name": "Interleukin-2 (cryptic helix-α1 site)",
        "category": "cryptic",
        "apo_pdb": "1M47", "apo_chain": "A",
        "holo_pdb": "1M49", "holo_chain": "A",
        "extra_exclude": frozenset(),
        "citation": "Braisted 2003; Arkin 2003",
    },
    {
        "id": "GCK",
        "name": "Glucokinase (allosteric activator site)",
        "category": "cryptic",
        "apo_pdb": "1V4S", "apo_chain": "A",
        "holo_pdb": "3IMX", "holo_chain": "A",  # compound B84; 1V4T has no activator
        "extra_exclude": frozenset({"GLC", "FRU", "ATP", "ADP", "AMP"}),
        "citation": "Kamata 2004; Zhi 2010",
    },
    {
        "id": "p38_DFGout",
        "name": "p38α MAPK DFG-out pocket (BIRB 796)",
        "category": "cryptic",
        "apo_pdb": "1P38", "apo_chain": "A",
        "holo_pdb": "2ZB1", "holo_chain": "A",
        "extra_exclude": frozenset(),
        "citation": "Pargellis 2002; Regan 2003",
    },
    {
        "id": "HIVRT_NNRTI",
        "name": "HIV-1 RT NNRTI binding pocket (nevirapine)",
        "category": "cryptic",
        "apo_pdb": "1HMV", "apo_chain": "A",
        "holo_pdb": "1RTH", "holo_chain": "A",
        "extra_exclude": frozenset({"AZT", "TMP", "MG"}),
        "max_residues": 500,  # heterodimer >900 res; skip to keep runtime sane
        "citation": "Kohlstaedt 1992; Ren 1995",
    },
    {
        "id": "SRC_myristate",
        "name": "Src kinase myristate/SH2-linker pocket",
        "category": "cryptic",
        "apo_pdb": "2SRC", "apo_chain": "A",
        "holo_pdb": "3EL8", "holo_chain": "A",
        "extra_exclude": frozenset({"MYR", "ADP", "ATP", "ANP"}),
        "citation": "Cowan-Jacob 2005; Shekhar 2009",
    },
    {
        "id": "MDM2",
        "name": "MDM2 p53-binding cleft (cryptic Trp/Leu pocket)",
        "category": "cryptic",
        "apo_pdb": "1Z1M", "apo_chain": "A",
        "holo_pdb": "4HBM", "holo_chain": "A",  # 1T4F had no ligand; 4HBM=nutlin-3
        "extra_exclude": frozenset(),
        "citation": "Vassilev 2004; Kussie 1996",
    },

    # ── CONFORMATIONAL / ALLOSTERIC ───────────────────────────────────────────
    # Large-scale movement required; pocket present but re-shaped in holo.

    {
        "id": "AK",
        "name": "Adenylate kinase (open→closed substrate sites)",
        "category": "conformational",
        "apo_pdb": "4AKE", "apo_chain": "A",  # open/unbound form
        "holo_pdb": "1AKE", "holo_chain": "A",  # closed form with AP5A; was swapped
        "extra_exclude": frozenset({"AMP", "ADP", "MG"}),  # pick AP5, not AMP
        "citation": "Muller 1996",
    },
    {
        "id": "CypA",
        "name": "Cyclophilin A (active-site cryptic sub-pocket)",
        "category": "conformational",
        "apo_pdb": "1OCA", "apo_chain": "A",  # apo CypA
        "holo_pdb": "2CPL", "holo_chain": "A",  # cyclosporin A bound
        "extra_exclude": frozenset(),
        "citation": "Ke 1994; Kallen 1991",
    },

    # ── ORTHOSTERIC CONTROLS ──────────────────────────────────────────────────
    # Pocket always clearly visible in apo structure; both tools should pass.

    {
        "id": "lysozyme",
        "name": "Hen lysozyme (active site, orthosteric)",
        "category": "orthosteric",
        "apo_pdb": "1HEL", "apo_chain": "A",
        "known_residues": {35, 52, 101, 102, 103, 104, 107, 108},
        "citation": "Blake 1965",
    },
    {
        "id": "HIV_PR",
        "name": "HIV-1 protease (active site flap region)",
        "category": "orthosteric",
        "apo_pdb": "1HPV", "apo_chain": "A",
        "known_residues": {25, 26, 27, 28, 29, 30, 49, 50, 51, 52, 53},
        "citation": "Lapatto 1989",
    },
    {
        "id": "thrombin",
        "name": "Thrombin (active site S1/S2 pockets)",
        "category": "orthosteric",
        "apo_pdb": "2RGL", "apo_chain": "A",  # 1HGT had hirudin; 2RGL is clean apo
        "holo_pdb": "1TOM", "holo_chain": "H",  # MIN (melagatran) is in chain H
        "extra_exclude": frozenset({"TYS"}),
        "citation": "Stubbs 1990",
    },
    {
        "id": "trypsin",
        "name": "Trypsin (S1 active site)",
        "category": "orthosteric",
        "apo_pdb": "1S0Q", "apo_chain": "A",
        "holo_pdb": "3PTB", "holo_chain": "A",  # benzamidine (BEN); 1PPH has BPTI protein
        "extra_exclude": frozenset({"CA"}),
        "citation": "Walter 1982; Marquart 1983",
    },
    {
        "id": "DHFR",
        "name": "DHFR (folate/MTX binding site)",
        "category": "orthosteric",
        "apo_pdb": "7DFR", "apo_chain": "A",
        "holo_pdb": "4DFR", "holo_chain": "A",
        "extra_exclude": frozenset({"NADP", "NAP", "NAI"}),
        "citation": "Bolin 1982",
    },
]


# ── PDB parsing helpers ────────────────────────────────────────────────────────

def download_pdb(pdb_id: str, dest_dir: Path) -> Path:
    out = dest_dir / f"{pdb_id}.pdb"
    if out.exists():
        return out
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  Downloading {pdb_id}...", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, out)
        print("done")
    except Exception as e:
        print(f"FAILED ({e})")
        if out.exists():
            out.unlink()
        raise
    return out


def _parse_atoms(pdb_path: Path) -> list[dict]:
    """Return list of dicts with keys: record, chain, resname, resseq, x, y, z."""
    atoms = []
    with open(pdb_path, errors="replace") as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ("ATOM", "HETATM"):
                continue
            try:
                atoms.append({
                    "record": rec,
                    "name": line[12:16].strip(),
                    "resname": line[17:20].strip(),
                    "chain": line[21].strip(),
                    "resseq": int(line[22:26].strip()),
                    "x": float(line[30:38]),
                    "y": float(line[38:46]),
                    "z": float(line[46:54]),
                })
            except (ValueError, IndexError):
                pass
    return atoms


def extract_binding_site(holo_path: Path, holo_chain: str,
                          extra_exclude: frozenset = frozenset()) -> set[int]:
    """Find principal ligand in holo structure; return residue numbers of
    protein residues within HOLO_CUTOFF Å of any ligand atom."""
    atoms = _parse_atoms(holo_path)
    exclude = SOLVENT_CODES | extra_exclude

    # Collect HETATM groups (resname, chain, resseq) — skip solvent
    from collections import defaultdict
    lig_groups: dict[tuple, list] = defaultdict(list)
    for a in atoms:
        if a["record"] == "HETATM" and a["resname"] not in exclude:
            key = (a["resname"], a["chain"], a["resseq"])
            lig_groups[key].append(a)

    if not lig_groups:
        return set()

    # Pick the group with the most atoms (principal ligand)
    principal_key = max(lig_groups, key=lambda k: len(lig_groups[k]))
    lig_atoms = lig_groups[principal_key]
    lig_coords = [(a["x"], a["y"], a["z"]) for a in lig_atoms]
    print(f"    Ligand: {principal_key[0]} chain {principal_key[1]} "
          f"({len(lig_atoms)} atoms)")

    # Find protein residues within cutoff
    prot_atoms = [a for a in atoms if a["record"] == "ATOM"]
    binding: set[int] = set()
    cutoff2 = HOLO_CUTOFF ** 2
    for pa in prot_atoms:
        if any(
            (pa["x"] - lx) ** 2 + (pa["y"] - ly) ** 2 + (pa["z"] - lz) ** 2 <= cutoff2
            for lx, ly, lz in lig_coords
        ):
            binding.add(pa["resseq"])

    return binding


# ── overlap metric ─────────────────────────────────────────────────────────────

def residue_overlap(cluster_residues: list[str], known: set[int]) -> float:
    found: set[int] = set()
    for label in cluster_residues:
        try:
            found.add(int("".join(c for c in label.split(":")[0] if c.isdigit())))
        except (ValueError, IndexError):
            pass
    return len(found & known) / len(known) if known else 0.0


# ── Lacuna runner ──────────────────────────────────────────────────────────────

def run_lacuna(pdb_path: Path, n_conformers: int) -> tuple[list, float]:
    from lacuna.io.structure import load_structure, coords_array
    from lacuna.ensemble.random_backend import RandomBackend
    from lacuna.pockets.detector import detect_pockets
    from lacuna.pockets.clusterer import cluster_pockets

    structure = load_structure(pdb_path)
    backend = RandomBackend(seed=42)

    t0 = time.perf_counter()
    coord_sets = backend.generate(pdb_path, n_conformers=n_conformers)
    base = coords_array(structure)
    all_coords = [base] + coord_sets

    pocket_lists = []
    for ci, coords in enumerate(all_coords):
        pockets = detect_pockets(coords, structure)
        for p in pockets:
            p.conformer_idx = ci
        pocket_lists.append(pockets)

    clusters = cluster_pockets(pocket_lists, n_conformers=len(all_coords))
    elapsed = time.perf_counter() - t0
    return clusters, elapsed


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformers", type=int, default=CONFORMERS)
    parser.add_argument("--quick", action="store_true",
                        help="Use 10 conformers for a faster run")
    parser.add_argument("--category", choices=["cryptic", "conformational",
                                                "orthosteric", "all"],
                        default="all")
    args = parser.parse_args()

    n_conf = 10 if args.quick else args.conformers

    pdb_dir = Path(__file__).parent / "pdb_cache"
    pdb_dir.mkdir(exist_ok=True)

    entries = DATASET if args.category == "all" else [
        e for e in DATASET if e["category"] == args.category
    ]

    print("=" * 70)
    print(f"  LACUNA — CRYPTIC POCKET BENCHMARK  ({len(entries)} proteins, {n_conf} conformers)")
    print("=" * 70)

    results = []

    for entry in entries:
        print(f"\n{'─'*70}")
        print(f"  [{entry['category'].upper()}]  {entry['id']}  —  {entry['name']}")
        print(f"{'─'*70}")

        # ── resolve binding site ──────────────────────────────────────────────
        known: set[int] = set()

        if "known_residues" in entry:
            known = entry["known_residues"]
            print(f"  Binding site: {len(known)} residues (literature-defined)")
        else:
            try:
                holo_path = download_pdb(entry["holo_pdb"], pdb_dir)
            except Exception:
                print("  [SKIP] holo download failed")
                results.append({**entry, "status": "skip_holo_download"})
                continue
            known = extract_binding_site(
                holo_path,
                entry.get("holo_chain", "A"),
                entry.get("extra_exclude", frozenset()),
            )
            if not known:
                print("  [SKIP] no principal ligand found in holo structure")
                results.append({**entry, "status": "skip_no_ligand"})
                continue
            print(f"  Binding site: {len(known)} residues (auto-extracted, {HOLO_CUTOFF}Å cutoff)")

        # ── run Lacuna on apo ─────────────────────────────────────────────────
        try:
            apo_path = download_pdb(entry["apo_pdb"], pdb_dir)
        except Exception:
            print("  [SKIP] apo download failed")
            results.append({**entry, "status": "skip_apo_download"})
            continue

        # Guard against very large multi-chain complexes that would take minutes
        max_res = entry.get("max_residues", 600)
        try:
            from lacuna.io.structure import load_structure
            s = load_structure(apo_path)
            if len(s.residues) > max_res:
                print(f"  [SKIP] {len(s.residues)} residues > max_residues={max_res} "
                      f"(use --max-residues to override)")
                results.append({**entry, "status": "skip_too_large",
                                "n_residues": len(s.residues)})
                continue
        except Exception:
            pass

        try:
            clusters, elapsed = run_lacuna(apo_path, n_conf)
        except Exception as e:
            print(f"  [ERROR] Lacuna failed: {e}")
            results.append({**entry, "status": "error", "error": str(e)})
            continue

        # ── score ─────────────────────────────────────────────────────────────
        best_ov, best_rank = 0.0, None
        for c in clusters[:5]:
            ov = residue_overlap(c.lining_residues, known)
            if ov > best_ov:
                best_ov, best_rank = ov, c.rank

        found = best_ov >= OVERLAP_THRESHOLD
        status = "PASS" if found else "MISS"
        marker = "✅" if found else "❌"
        print(f"  {marker} {status}  overlap={best_ov:.0%}  rank={best_rank}  "
              f"clusters={len(clusters)}  time={elapsed:.1f}s")

        results.append({
            "id": entry["id"],
            "name": entry["name"],
            "category": entry["category"],
            "apo_pdb": entry["apo_pdb"],
            "status": status,
            "overlap": round(best_ov, 3),
            "rank": best_rank,
            "n_clusters": len(clusters),
            "elapsed_s": round(elapsed, 2),
            "n_known_residues": len(known),
        })

    # ── summary ───────────────────────────────────────────────────────────────
    ran = [r for r in results if r["status"] in ("PASS", "MISS")]
    skipped = [r for r in results if r["status"] not in ("PASS", "MISS")]

    print(f"\n{'='*70}")
    print("  RESULTS BY CATEGORY")
    print(f"{'='*70}")

    categories = ["cryptic", "conformational", "orthosteric"]
    grand_pass = grand_total = 0

    for cat in categories:
        cat_rows = [r for r in ran if r.get("category") == cat]
        if not cat_rows:
            continue
        n_pass = sum(1 for r in cat_rows if r["status"] == "PASS")
        print(f"\n  {cat.upper()}  ({n_pass}/{len(cat_rows)})")
        for r in cat_rows:
            m = "✅" if r["status"] == "PASS" else "❌"
            ov = f"{r['overlap']:.0%}" if r.get("overlap") is not None else "—"
            rk = f"@{r['rank']}" if r.get("rank") else ""
            t = f"{r['elapsed_s']:.1f}s" if r.get("elapsed_s") is not None else ""
            print(f"    {m} {r['apo_pdb']}  {ov:5s} {rk:4s}  {t:6s}  {r['name']}")
        grand_pass += n_pass
        grand_total += len(cat_rows)

    print(f"\n{'─'*70}")
    print(f"  TOTAL (cryptic + conformational + orthosteric): {grand_pass}/{grand_total}")
    if skipped:
        print(f"  Skipped: {len(skipped)} "
              f"({', '.join(s['id'] for s in skipped)})")

    # Save JSON
    out_path = Path(__file__).parent / "cryptic_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results → {out_path}")


if __name__ == "__main__":
    main()
