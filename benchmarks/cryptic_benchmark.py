"""Cryptic pocket benchmark - apo/holo PDB pairs.

For each entry we either supply known binding-site residues directly (taken
from the published literature) OR supply a holo PDB ID and let the script
auto-extract the binding site as protein residues within HOLO_CUTOFF Å of
the principal ligand.

Primary success criterion: any top-5 Lacuna pocket centroid within
CENTROID_THRESHOLD Å of the reference binding-site centroid.  This matches
the field-standard metric (e.g. CryptoSite, CrypticOpen) and is insensitive
to residue-numbering offsets across crystal structures.

Secondary metric reported: residue overlap ≥ OVERLAP_THRESHOLD (kept for
backward compatibility and direct comparison with earlier results).

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

# Ensure UTF-8 output on Windows terminals that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── constants ─────────────────────────────────────────────────────────────────

# When citing results produced by this benchmark, please use:
#   Moore CW (2026) Lacuna: Cryptic Binding Pocket Discovery via Conformational
#   Ensemble Analysis. https://github.com/mooreneural/lacuna
TOOL_CITATION = "Moore 2026 Lacuna (github.com/mooreneural/lacuna)"

CONFORMERS = 20
HOLO_CUTOFF = 4.5          # Å - residue within this of any ligand atom → binding site
CENTROID_THRESHOLD = 4.0   # Å - primary: pocket centroid within this of site centroid
OVERLAP_THRESHOLD = 0.30   # legacy recall = |found∩known|/|known| (size-gameable)
JACCARD_THRESHOLD = 0.25   # size-robust IoU = |found∩known|/|found∪known|
CORE_RADIUS = 8.0          # Å - hotspot-core: known Cα within this of pocket hotspot
HOTSPOT_CORE_THRESHOLD = 0.50  # size-robust: fraction of site wrapping the hotspot

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
        "citation": "Eriksson 1992; Mobley 2007; Moore 2026 Lacuna",
    },
    {
        "id": "KRAS_SIIP",
        "name": "K-Ras WT apo (switch-II cryptic pocket)",
        "category": "cryptic",
        "apo_pdb": "4OBE", "apo_chain": "A",
        "known_residues": {12, 13, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
        "citation": "Ostrem 2013; Moore 2026 Lacuna",
    },
    {
        "id": "IL2",
        "name": "Interleukin-2 (cryptic helix-α1 site)",
        "category": "cryptic",
        "apo_pdb": "1M47", "apo_chain": "A",
        "holo_pdb": "1M49", "holo_chain": "A",
        "extra_exclude": frozenset(),
        "citation": "Braisted 2003; Arkin 2003; Moore 2026 Lacuna",
    },
    {
        "id": "GCK",
        "name": "Glucokinase (allosteric activator site)",
        "category": "cryptic",
        "apo_pdb": "1V4S", "apo_chain": "A",
        "holo_pdb": "3IMX", "holo_chain": "A",  # compound B84; 1V4T has no activator
        "extra_exclude": frozenset({"GLC", "FRU", "ATP", "ADP", "AMP"}),
        "citation": "Kamata 2004; Zhi 2010; Moore 2026 Lacuna",
    },
    {
        "id": "p38_DFGout",
        "name": "p38α MAPK DFG-out pocket (BIRB 796)",
        "category": "cryptic",
        "apo_pdb": "1P38", "apo_chain": "A",
        "holo_pdb": "2ZB1", "holo_chain": "A",
        "extra_exclude": frozenset(),
        "citation": "Pargellis 2002; Regan 2003; Moore 2026 Lacuna",
    },
    {
        "id": "HIVRT_NNRTI",
        "name": "HIV-1 RT NNRTI binding pocket (nevirapine)",
        "category": "cryptic",
        "apo_pdb": "1HMV", "apo_chain": "A",
        "holo_pdb": "1RTH", "holo_chain": "A",
        "extra_exclude": frozenset({"AZT", "TMP", "MG"}),
        "max_residues": 600,  # chain A = 536 res; was 500, raised to allow it to run
        "citation": "Kohlstaedt 1992; Ren 1995; Moore 2026 Lacuna",
    },
    {
        "id": "SRC_myristate",
        "name": "Src kinase myristate/SH2-linker pocket",
        "category": "cryptic",
        "apo_pdb": "2SRC", "apo_chain": "A",
        "holo_pdb": "3EL8", "holo_chain": "A",
        "extra_exclude": frozenset({"MYR", "ADP", "ATP", "ANP"}),
        "citation": "Cowan-Jacob 2005; Shekhar 2009; Moore 2026 Lacuna",
    },
    {
        "id": "MDM2",
        "name": "MDM2 p53-binding cleft (cryptic Trp/Leu pocket)",
        "category": "cryptic",
        "apo_pdb": "1Z1M", "apo_chain": "A",
        "holo_pdb": "4HBM", "holo_chain": "A",  # 1T4F had no ligand; 4HBM=nutlin-3
        "extra_exclude": frozenset(),
        "citation": "Vassilev 2004; Kussie 1996; Moore 2026 Lacuna",
    },

    {
        "id": "SHP2_allosteric",
        "name": "SHP-2 allosteric tunnel (SHP836/SHP099 site)",
        "category": "cryptic",
        # 2SHP = autoinhibited SHP-2; N-SH2 occludes the allosteric tunnel.
        # 5EHP = SHP-2 with SHP836 allosteric inhibitor (46 atoms) bound at the
        # N-SH2/C-SH2/PTP junction.  A homodimer in the crystal; we use chain A only.
        "apo_pdb": "2SHP", "apo_chain": "A",
        "holo_pdb": "5EHP", "holo_chain": "A",
        "extra_exclude": frozenset(),
        "citation": "Chen 2016 Nature; Tonks 2006 Nature Rev Mol Cell Biol; Moore 2026 Lacuna",
    },
    {
        "id": "ABL1_myristate",
        "name": "c-ABL myristate/allosteric pocket (asciminib target)",
        "category": "cryptic",
        # 3CS9 = ABL1 kinase domain + nilotinib (ATP site); myristate pocket is empty.
        # 2FO0 = ABL1 SH3-SH2-kinase with MYR (myristic acid) in the C-lobe pocket.
        # extra_exclude removes P16 so MYR (15 atoms) becomes the principal ligand.
        "apo_pdb": "3CS9", "apo_chain": "A",
        "holo_pdb": "2FO0", "holo_chain": "A",
        "extra_exclude": frozenset({"NIL", "P16", "SEP"}),
        "citation": "Nagar 2002 Cell; Wylie 2017 Nature (asciminib); Moore 2026 Lacuna",
    },
    {
        "id": "PTP1B_allosteric",
        "name": "PTP1B allosteric site (C-terminal helix pocket)",
        "category": "cryptic",
        # 1A5Y = apo PTP1B; allosteric site at the C-terminal helix is unoccupied.
        # 2F6V = PTP1B with SK2 allosteric benzofuran inhibitor (29 atoms).
        "apo_pdb": "1A5Y", "apo_chain": "A",
        "holo_pdb": "2F6V", "holo_chain": "A",
        "extra_exclude": frozenset(),
        "citation": "Wiesmann 2004 Nature; Bhatt 2007 J Med Chem; Moore 2026 Lacuna",
    },
    {
        "id": "NS5B_thumb",
        "name": "HCV NS5B thumb-site I allosteric pocket",
        "category": "cryptic",
        # 1NB4 = apo NS5B RdRp; priming loop occludes the thumb allosteric site.
        # 2I1R = NS5B with VXR thumb-site I inhibitor (56 atoms).
        "apo_pdb": "1NB4", "apo_chain": "A",
        "holo_pdb": "2I1R", "holo_chain": "A",
        "extra_exclude": frozenset(),
        "citation": "Koch 2006 J Biol Chem; Boyce 2009 PNAS; Moore 2026 Lacuna",
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
        "citation": "Muller 1996; Moore 2026 Lacuna",
    },
    {
        "id": "CypA",
        "name": "Cyclophilin A (active-site cryptic sub-pocket)",
        "category": "conformational",
        "apo_pdb": "1OCA", "apo_chain": "A",  # apo CypA
        "holo_pdb": "2CPL", "holo_chain": "A",  # cyclosporin A bound
        "extra_exclude": frozenset(),
        "citation": "Ke 1994; Kallen 1991; Moore 2026 Lacuna",
    },

    # ── ORTHOSTERIC CONTROLS ──────────────────────────────────────────────────
    # Pocket always clearly visible in apo structure; both tools should pass.

    {
        "id": "lysozyme",
        "name": "Hen lysozyme (active site, orthosteric)",
        "category": "orthosteric",
        "apo_pdb": "1HEL", "apo_chain": "A",
        "known_residues": {35, 52, 101, 102, 103, 104, 107, 108},
        "citation": "Blake 1965; Moore 2026 Lacuna",
    },
    {
        "id": "HIV_PR",
        "name": "HIV-1 protease (active site flap region)",
        "category": "orthosteric",
        "apo_pdb": "1HPV", "apo_chain": "A",
        "known_residues": {25, 26, 27, 28, 29, 30, 49, 50, 51, 52, 53},
        "citation": "Lapatto 1989; Moore 2026 Lacuna",
    },
    {
        "id": "thrombin",
        "name": "Thrombin (active site S1/S2 pockets)",
        "category": "orthosteric",
        "apo_pdb": "2RGL", "apo_chain": "A",  # 1HGT had hirudin; 2RGL is clean apo
        "holo_pdb": "1TOM", "holo_chain": "H",  # MIN (melagatran) is in chain H
        "extra_exclude": frozenset({"TYS"}),
        "citation": "Stubbs 1990; Moore 2026 Lacuna",
    },
    {
        "id": "trypsin",
        "name": "Trypsin (S1 active site)",
        "category": "orthosteric",
        # NOTE: 1S0Q uses non-standard sequential numbering (660-882) that is
        # incompatible with 3PTB's chymotrypsin numbering (16-245).  Residue-overlap
        # scores 0% and centroid fails (reference residues not found in 1S0Q).
        # Use known_residues in 1S0Q's own numbering once the mapping is verified;
        # for now this entry documents the limitation rather than contributing to stats.
        "apo_pdb": "1S0Q", "apo_chain": "A",
        "holo_pdb": "3PTB", "holo_chain": "A",  # benzamidine (BEN); 1PPH has BPTI protein
        "extra_exclude": frozenset({"CA"}),
        "citation": "Walter 1982; Marquart 1983; Moore 2026 Lacuna",
    },
    {
        "id": "DHFR",
        "name": "DHFR (folate/MTX binding site)",
        "category": "orthosteric",
        "apo_pdb": "7DFR", "apo_chain": "A",
        "holo_pdb": "4DFR", "holo_chain": "A",
        "extra_exclude": frozenset({"NADP", "NAP", "NAI"}),
        "citation": "Bolin 1982; Moore 2026 Lacuna",
    },

    # ── ADDITIONAL CRYPTIC POCKETS (round 2) ──────────────────────────────────

    {
        "id": "BCLXL",
        "name": "BCL-XL BH3-binding groove (navitoclax/ABT-737)",
        "category": "cryptic",
        "apo_pdb": "1LXL", "apo_chain": "A",  # C-terminal helix occludes groove
        "holo_pdb": "2YXJ", "holo_chain": "A",  # N3C = ABT-737 (112 atoms)
        "extra_exclude": frozenset(),
        "citation": "Oltersdorf 2005; Tse 2008; Moore 2026 Lacuna",
    },
    {
        "id": "HIF2a",
        "name": "HIF-2α PAS-B internal cavity (belzutifan/PT2385)",
        # NOTE: 3F1O already contains endogenous ligand 2XY (20 atoms) in the cavity,
        # so the pocket is pre-opened in the input structure.  Reclassified as
        # orthosteric positive control; it should NOT count toward the cryptic pass rate.
        "category": "orthosteric",
        "apo_pdb": "3F1O", "apo_chain": "A",
        "holo_pdb": "5TBM", "holo_chain": "A",  # 79A = PT2385 (26 atoms); FDA-approved
        "extra_exclude": frozenset({"2XY"}),
        "citation": "Scheuermann 2009; Courtney 2018; Moore 2026 Lacuna",
    },
    {
        "id": "CASP1",
        "name": "Caspase-1 allosteric dimer-interface pocket",
        "category": "cryptic",
        "apo_pdb": "2HBQ", "apo_chain": "A",  # active site inhibitor PHQ; allosteric site empty
        "holo_pdb": "3NKT", "holo_chain": "A",  # 1HN = allosteric inhibitor (14 atoms)
        "extra_exclude": frozenset(),
        "citation": "Scheer 2006; Datta 2008; Moore 2026 Lacuna",
    },
    {
        "id": "ERK2",
        "name": "ERK2 allosteric binding site",
        "category": "cryptic",
        "apo_pdb": "2ERK", "apo_chain": "A",  # active phospho-ERK2, no allosteric ligand
        "holo_pdb": "4QTA", "holo_chain": "A",  # 38Z = allosteric inhibitor (44 atoms)
        "extra_exclude": frozenset(),
        "citation": "Hancock 2015; Moore 2026 Lacuna",
    },

    # ── ADDITIONAL CRYPTIC POCKETS (round 3) - targets N ≥ 20 ────────────────

    {
        "id": "BCL2_BH3",
        "name": "BCL-2 BH3-binding groove (venetoclax/ABT-199)",
        "category": "cryptic",
        # 1G5M = apo BCL-2 isoform 1 - no small-molecule HETATM, BH3 groove
        # partially occluded by the C-terminal flexible loop (cf. BCL-XL in 1LXL).
        # 6O0K = BCL-2 co-crystallised with venetoclax (LBM, 55 atoms).
        "apo_pdb": "1G5M", "apo_chain": "A",
        "holo_pdb": "6O0K", "holo_chain": "A",
        "extra_exclude": frozenset(),
        "citation": "Tse 2008 Cancer Cell; Souers 2013 Nat Med (venetoclax FDA-approved 2016); Moore 2026 Lacuna",
    },
    {
        "id": "IDH1_R132H",
        "name": "IDH1 R132H allosteric dimer-interface (ivosidenib target)",
        "category": "cryptic",
        # 3MAP = IDH1 R132H homodimer with NADP+/isocitrate substrates but NO
        # allosteric inhibitor - dimer-interface pocket is absent/closed.
        # 4UMX = IDH1 R132H + CPD-1 allosteric inhibitor (VVS, 27 atoms) at the
        # dimer interface.  NAP (NADP+, 96 atoms) must be excluded so VVS wins
        # the principal-ligand selection.
        "apo_pdb": "3MAP", "apo_chain": "A",
        "holo_pdb": "4UMX", "holo_chain": "A",
        "extra_exclude": frozenset({"NAP", "ICT"}),
        "citation": "Ward 2010 Nature; Rohle 2013 Science (ivosidenib FDA-approved 2018); Moore 2026 Lacuna",
    },
    {
        "id": "PKM2_activator",
        "name": "PKM2 allosteric activator pocket (TEPP-46 / subunit interface)",
        "category": "cryptic",
        # 1ZJH = human PKM2 apo (2005, Dombrauckas) - no HETATM at all; activator
        # pocket at the dimer-dimer interface is absent in this T-state structure.
        # 3U2Z = activator-bound PKM2 R-state (Anastasiou 2012).  Exclude FBP and
        # oxalate so the synthetic activator compound (residue "551", 100 atoms)
        # becomes the principal ligand for binding-site extraction.
        "apo_pdb": "1ZJH", "apo_chain": "A",
        "holo_pdb": "3U2Z", "holo_chain": "A",
        "extra_exclude": frozenset({"FBP", "OXL", "OXA", "AMP", "ADP", "ATP"}),
        "citation": "Anastasiou 2012 Cell (TEPP-46 activator class); Moore 2026 Lacuna",
    },
    {
        "id": "PPARG_allosteric",
        "name": "PPARγ LBD allosteric helix-12 site (metaglidasen/MBX-102)",
        "category": "cryptic",
        # 2PRG = PPARγ LBD with rosiglitazone at the canonical TZD agonist site;
        # the surface AF-2/helix-12 allosteric pocket is unoccupied.
        # 4PVU = PPARγ LBD + MBX-102 (MGZ, 22 atoms) at the allosteric site.
        # Exclude rosiglitazone (RSG) so MGZ wins principal-ligand selection.
        "apo_pdb": "2PRG", "apo_chain": "A",
        "holo_pdb": "4PVU", "holo_chain": "A",
        "extra_exclude": frozenset({"RSG", "RLX"}),
        "citation": "Nettles 2008 PNAS; Bruning 2007 Structure (metaglidasen Phase 3); Moore 2026 Lacuna",
    },
    {
        "id": "MMP13_allosteric",
        "name": "MMP-13 non-zinc allosteric S1′ tunnel",
        "category": "cryptic",
        # 2OZR = MMP-13 catalytic domain with a zinc-chelating hydroxamate inhibitor
        # (GG1, 256 atoms) at the orthosteric zinc site; the allosteric S1′ tunnel
        # is empty and unformed in this structure.
        # 3I7G = MMP-13 with a non-zinc-chelating allosteric inhibitor that opens
        # the S1′ tunnel (Engel 2005 paradigm, confirmed by Becker 2010).
        "apo_pdb": "2OZR", "apo_chain": "A",
        "holo_pdb": "3I7G", "holo_chain": "A",
        "extra_exclude": frozenset({"GG1", "HAE", "ZN", "CA"}),
        "citation": "Engel 2005 J Med Chem; Becker 2010 Nat Chem Biol; Moore 2026 Lacuna",
    },

    # ── ADDITIONAL VERIFIED CRYPTIC PAIRS (RCSB-confirmed apo/holo) ───────────
    {
        "id": "TEM1_allosteric",
        "name": "TEM-1 β-lactamase cryptic allosteric site (CBT)",
        "category": "cryptic",
        # 1JWP = TEM-1 (M182T) apo; the H11/H12 allosteric cryptic pocket is closed.
        # 1PZO = TEM-1 with CBT core-disrupting allosteric inhibitor at that site
        # (RCSB title: "…in Complex with a Novel, Core-Disrupting, Allosteric…").
        "apo_pdb": "1JWP", "apo_chain": "A",
        "holo_pdb": "1PZO", "holo_chain": "A",
        "extra_exclude": frozenset(),
        "citation": "Horn 2004 J Mol Biol; Bowman 2012 J Am Chem Soc; Moore 2026 Lacuna",
    },
    {
        "id": "RICIN_pterin",
        "name": "Ricin A-chain cryptic specificity pocket (pteroic acid)",
        "category": "cryptic",
        # 1RTC = ricin A-chain apo; the secondary pterin pocket beside the active
        # site is unformed. 1BR6 = ricin A-chain complexed with pteroic acid (PT1).
        "apo_pdb": "1RTC", "apo_chain": "A",
        "holo_pdb": "1BR6", "holo_chain": "A",
        "extra_exclude": frozenset(),
        "citation": "Mlsna 1993 Protein Sci; Yan 1997 J Mol Biol; Moore 2026 Lacuna",
    },
]


# ── opening-mechanism labels ───────────────────────────────────────────────────
# Coarse, literature-based annotation of the DOMINANT structural change that opens
# each cryptic site (Beglov 2018; CryptoBench 2024 examples; per-target papers).
# The taxonomy follows the mechanism classes discussed for cryptic pockets:
#   sidechain - a side-chain rotamer flip unblocks the site
#   loop      - a loop swings/reorders over the site
#   helix     - helix / secondary-structure remodeling or a capping helix moves
#   hinge     - large domain / inter-lobe breathing
#   interface - an oligomeric (inter-subunit) interface pocket
# These are single dominant labels for a genuinely multi-factor process, and each
# class has a small N, so per-mechanism rates are diagnostic, not statistical.
MECHANISM = {
    # cryptic
    "T4L_L99A": "sidechain", "KRAS_SIIP": "loop", "IL2": "sidechain",
    "GCK": "hinge", "p38_DFGout": "loop", "HIVRT_NNRTI": "sidechain",
    "SRC_myristate": "helix", "MDM2": "sidechain", "SHP2_allosteric": "hinge",
    "ABL1_myristate": "helix", "PTP1B_allosteric": "helix", "NS5B_thumb": "loop",
    "BCLXL": "helix", "CASP1": "interface", "ERK2": "loop", "BCL2_BH3": "helix",
    "IDH1_R132H": "interface", "PKM2_activator": "interface",
    "PPARG_allosteric": "helix", "MMP13_allosteric": "loop",
    "TEM1_allosteric": "helix", "RICIN_pterin": "loop",
    # conformational
    "AK": "hinge", "CypA": "sidechain",
    # orthosteric controls (always-open; mechanism is nominal)
    "lysozyme": "sidechain", "HIV_PR": "loop", "thrombin": "loop",
    "trypsin": "sidechain", "DHFR": "loop", "HIF2a": "sidechain",
}


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


def extract_binding_site(
    holo_path: Path,
    holo_chain: str,
    extra_exclude: frozenset = frozenset(),
) -> tuple[set[int], tuple[float, float, float] | None]:
    """Find principal ligand in holo structure.

    Returns:
        (binding_residues, ligand_centroid) - binding_residues is the set of
        protein residue numbers within HOLO_CUTOFF Å of the principal ligand;
        ligand_centroid is the mean (x,y,z) of all ligand atoms.  Both are None
        / empty if no principal ligand is found.
    """
    atoms = _parse_atoms(holo_path)
    exclude = SOLVENT_CODES | extra_exclude

    # Collect HETATM groups (resname, chain, resseq) - skip solvent
    from collections import defaultdict
    lig_groups: dict[tuple, list] = defaultdict(list)
    for a in atoms:
        if a["record"] == "HETATM" and a["resname"] not in exclude:
            key = (a["resname"], a["chain"], a["resseq"])
            lig_groups[key].append(a)

    if not lig_groups:
        return set(), None

    # Pick the group with the most atoms (principal ligand)
    principal_key = max(lig_groups, key=lambda k: len(lig_groups[k]))
    lig_atoms = lig_groups[principal_key]
    lig_coords = [(a["x"], a["y"], a["z"]) for a in lig_atoms]
    print(f"    Ligand: {principal_key[0]} chain {principal_key[1]} "
          f"({len(lig_atoms)} atoms)")

    # Ligand centroid
    n = len(lig_coords)
    centroid: tuple[float, float, float] = (
        sum(c[0] for c in lig_coords) / n,
        sum(c[1] for c in lig_coords) / n,
        sum(c[2] for c in lig_coords) / n,
    )

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

    return binding, centroid


def known_site_ca_coords(
    apo_path: Path,
    apo_chain: str,
    known_residues: set[int],
) -> list[tuple[float, float, float]]:
    """Return the Cα positions of the known binding-site residues in the apo frame."""
    atoms = _parse_atoms(apo_path)
    return [
        (a["x"], a["y"], a["z"])
        for a in atoms
        if a["record"] == "ATOM" and a["name"] == "CA"
        and a["chain"] == apo_chain and a["resseq"] in known_residues
    ]


def compute_known_site_centroid(
    apo_path: Path,
    apo_chain: str,
    known_residues: set[int],
) -> tuple[float, float, float] | None:
    """Return the mean Cα position of the known binding-site residues."""
    ca = known_site_ca_coords(apo_path, apo_chain, known_residues)
    if not ca:
        return None
    n = len(ca)
    return (
        sum(c[0] for c in ca) / n,
        sum(c[1] for c in ca) / n,
        sum(c[2] for c in ca) / n,
    )


def hotspot_core_overlap(
    pocket_centroid: tuple[float, float, float],
    known_ca: list[tuple[float, float, float]],
) -> float:
    """Size-robust, hotspot-anchored overlap.

    Fraction of the known-site residues whose Cα lies within ``CORE_RADIUS`` of
    the pocket's buriedness-weighted hotspot centroid. Because it depends only on
    the single hotspot point (not on how many residues the pocket lines) it cannot
    be inflated by returning a larger pocket, and because it is anchored at the
    buried core rather than the site centroid it tolerates elongated sites whose
    geometric centre falls in solvent. Complements Jaccard (set overlap) and the
    strict site-centroid distance.
    """
    if not known_ca:
        return 0.0
    cx, cy, cz = pocket_centroid
    r2 = CORE_RADIUS ** 2
    hits = sum(
        1 for (x, y, z) in known_ca
        if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r2
    )
    return hits / len(known_ca)


def pocket_min_centroid_dist(
    clusters: list,
    ref_centroid: tuple[float, float, float] | None,
    top_n: int = 5,
) -> tuple[float, int | None]:
    """Return (min_dist_Å, rank) of the closest top-N cluster centroid to ref_centroid."""
    if ref_centroid is None:
        return float("inf"), None
    rx, ry, rz = ref_centroid
    best_dist, best_rank = float("inf"), None
    for c in clusters[:top_n]:
        cx, cy, cz = c.centroid
        dist = ((cx - rx) ** 2 + (cy - ry) ** 2 + (cz - rz) ** 2) ** 0.5
        if dist < best_dist:
            best_dist, best_rank = dist, c.rank
    return best_dist, best_rank


# ── residue metrics ────────────────────────────────────────────────────────────

def _found_resnums(cluster_residues: list[str]) -> set[int]:
    """Parse residue sequence numbers out of Lacuna lining-residue labels.

    Labels are formatted ``NAME+seq:chain`` (e.g. ``ALA123:A``); we take the part
    before ``:`` and keep its digits.
    """
    found: set[int] = set()
    for label in cluster_residues:
        try:
            found.add(int("".join(c for c in label.split(":")[0] if c.isdigit())))
        except (ValueError, IndexError):
            pass
    return found


def residue_overlap(cluster_residues: list[str], known: set[int]) -> float:
    """Recall of the known site: |found ∩ known| / |known|.

    NOTE: size-gameable - a pocket with many lining residues trivially overlaps a
    small known site, so a model can inflate this by simply returning larger
    pockets.  Reported for backward comparison; prefer ``residue_jaccard`` (below)
    or the centroid criterion for a size-robust judgement.
    """
    found = _found_resnums(cluster_residues)
    return len(found & known) / len(known) if known else 0.0


def residue_jaccard(cluster_residues: list[str], known: set[int]) -> float:
    """Size-robust overlap (intersection-over-union): |found ∩ known| / |found ∪ known|.

    Unlike ``residue_overlap`` this penalises oversized pockets: adding spurious
    lining residues grows the union and lowers the score, so it cannot be gamed by
    ranking on raw pocket volume.
    """
    found = _found_resnums(cluster_residues)
    union = found | known
    return len(found & known) / len(union) if union else 0.0


# ── Lacuna runner ──────────────────────────────────────────────────────────────

def _make_backend(name: str, nma_rmsd: float = 2.0, nma_modes: int = 10,
                  boltz_msa: bool = False, openmm_temp: float = 310.0,
                  openmm_time: float = 50.0):
    if name == "nma":
        from lacuna.ensemble.nma_backend import NMABackend
        return NMABackend(seed=42, max_rmsd=nma_rmsd, n_modes=nma_modes)
    if name == "random":
        from lacuna.ensemble.random_backend import RandomBackend
        return RandomBackend(seed=42)
    if name == "openmm":
        from lacuna.ensemble.openmm_backend import OpenMMBackend
        return OpenMMBackend(temperature_k=openmm_temp, simulation_time_ps=openmm_time)
    if name == "boltz":
        from lacuna.ensemble.boltz_backend import BoltzBackend
        return BoltzBackend(use_msa_server=boltz_msa)
    raise ValueError(f"Unknown backend {name!r}")


def run_lacuna(
    pdb_path: Path,
    n_conformers: int,
    chain: str | None = None,
    backend_name: str = "nma",
    rank_by: str = "druggability",
    homodimer: bool = False,
    nma_rmsd: float = 2.0,
    nma_modes: int = 10,
    boltz_msa: bool = False,
    openmm_temp: float = 310.0,
    openmm_time: float = 50.0,
) -> tuple[list, float]:
    from lacuna.io.structure import load_structure, coords_array, make_biological_assembly
    from lacuna.io.writers import write_structure_pdb
    from lacuna.pockets.detector import detect_pockets
    from lacuna.pockets.clusterer import cluster_pockets

    backend = _make_backend(backend_name, nma_rmsd=nma_rmsd, nma_modes=nma_modes,
                            boltz_msa=boltz_msa, openmm_temp=openmm_temp,
                            openmm_time=openmm_time)
    t0 = time.perf_counter()

    if homodimer:
        # Dimer-interface pockets form between protomers and cannot be detected in a
        # single chain. Build the biological assembly (BIOMT) and analyze all chains.
        mono = load_structure(pdb_path, chain=None)
        assembly = make_biological_assembly(pdb_path, mono)
        if len(assembly.atoms) > len(mono.atoms):
            eff_path = pdb_path.with_name(pdb_path.stem + "_assembly.pdb")
            write_structure_pdb(assembly, eff_path)
            structure = assembly
        else:
            eff_path, structure = pdb_path, mono  # AU already multi-chain
        gen_chain = None
    else:
        structure = load_structure(pdb_path, chain=chain)
        eff_path, gen_chain = pdb_path, chain

    coord_sets = backend.generate(eff_path, n_conformers=n_conformers, chain=gen_chain)
    base = coords_array(structure)
    all_coords = [base] + coord_sets

    pocket_lists = []
    for ci, coords in enumerate(all_coords):
        pockets = detect_pockets(coords, structure)
        for p in pockets:
            p.conformer_idx = ci
        pocket_lists.append(pockets)

    clusters = cluster_pockets(pocket_lists, n_conformers=len(all_coords), rank_by=rank_by)
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
    parser.add_argument("--backend", choices=["nma", "random", "openmm", "boltz"],
                        default="nma",
                        help="Ensemble backend (default: nma - the package default)")
    parser.add_argument("--rank-by", dest="rank_by",
                        choices=["crypticity", "druggability", "persistence", "balanced"],
                        default="crypticity",
                        help="Pocket ranking strategy (default: crypticity)")
    parser.add_argument("--only", default=None,
                        help="Comma-separated entry IDs to run a subset "
                             "(e.g. --only p38_DFGout,SHP2_allosteric)")
    parser.add_argument("--top-n", dest="top_n", type=int, default=5,
                        help="Rank cutoff for a success (default: top-5). Use a "
                             "larger value to diagnose ranking vs detection.")
    parser.add_argument("--nma-rmsd", dest="nma_rmsd", type=float, default=2.0,
                        help="NMA max Cα RMSD amplitude in Å (default: 2.0)")
    parser.add_argument("--nma-modes", dest="nma_modes", type=int, default=10,
                        help="NMA number of low-frequency modes (default: 10)")
    parser.add_argument("--boltz-msa", dest="boltz_msa", action="store_true",
                        help="Boltz backend: fetch an MSA from the ColabFold server "
                             "(native-like structures) instead of msa:empty PLM-only mode")
    parser.add_argument("--openmm-temp", dest="openmm_temp", type=float, default=310.0,
                        help="OpenMM backend temperature in K (default 310; raise, e.g. "
                             "400, for enhanced-sampling cavity opening)")
    parser.add_argument("--openmm-time", dest="openmm_time", type=float, default=50.0,
                        help="OpenMM backend total production MD time per protein in ps "
                             "(default 50)")
    args = parser.parse_args()

    n_conf = 10 if args.quick else args.conformers

    pdb_dir = Path(__file__).parent / "pdb_cache"
    pdb_dir.mkdir(exist_ok=True)

    entries = DATASET if args.category == "all" else [
        e for e in DATASET if e["category"] == args.category
    ]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        entries = [e for e in entries if e["id"] in wanted]

    print("=" * 70)
    print(f"  LACUNA - CRYPTIC POCKET BENCHMARK  ({len(entries)} proteins, {n_conf} conformers)")
    print(f"  backend={args.backend}  rank_by={args.rank_by}  top_n={args.top_n}"
          f"  nma_rmsd={args.nma_rmsd}  nma_modes={args.nma_modes}")
    print("=" * 70)

    results = []

    for entry in entries:
        print(f"\n{'─'*70}")
        print(f"  [{entry['category'].upper()}]  {entry['id']}  -  {entry['name']}")
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
            known, _holo_centroid = extract_binding_site(
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

        # Reference centroid: Cα centroid of binding-site residues in the APO
        # structure.  Using apo coordinates (not holo) keeps both the reference
        # and the Lacuna pocket centroids in the same coordinate frame.
        ref_centroid = compute_known_site_centroid(
            apo_path, entry.get("apo_chain", "A"), known
        )
        known_ca = known_site_ca_coords(apo_path, entry.get("apo_chain", "A"), known)

        # Guard against very large multi-chain complexes that would take minutes
        apo_chain = entry.get("apo_chain")
        max_res = entry.get("max_residues", 600)
        try:
            from lacuna.io.structure import load_structure
            s = load_structure(apo_path, chain=apo_chain)
            if len(s.residues) > max_res:
                print(f"  [SKIP] {len(s.residues)} residues > max_residues={max_res} "
                      f"(use --max-residues to override)")
                results.append({**entry, "status": "skip_too_large",
                                "n_residues": len(s.residues)})
                continue
        except Exception:
            pass

        try:
            clusters, elapsed = run_lacuna(
                apo_path, n_conf, chain=apo_chain,
                backend_name=args.backend, rank_by=args.rank_by,
                homodimer=entry.get("homodimer", False),
                nma_rmsd=args.nma_rmsd, nma_modes=args.nma_modes,
                boltz_msa=args.boltz_msa,
                openmm_temp=args.openmm_temp, openmm_time=args.openmm_time,
            )
        except Exception as e:
            print(f"  [ERROR] Lacuna failed: {e}")
            results.append({**entry, "status": "error", "error": str(e)})
            continue

        # ── score ─────────────────────────────────────────────────────────────
        # Centroid distance: field-standard, robust to residue-numbering offsets.
        # Computed in the APO coordinate frame (ref centroid = Cα centroid of binding
        # site residues in the apo structure, not from the holo - see above).
        best_dist, best_dist_rank = pocket_min_centroid_dist(clusters, ref_centroid, top_n=args.top_n)

        # Residue metrics: recall (legacy, size-gameable) and Jaccard (size-robust).
        # Tracked independently so the best-overlap and best-Jaccard pockets may be
        # different clusters; best_ov_size is the lining-residue count of the
        # best-recall pocket, exposing size-gaming (a large found set inflates recall
        # while depressing Jaccard).
        best_ov, best_rank, best_ov_size = 0.0, None, 0
        best_jac, best_jac_rank = 0.0, None
        best_core = 0.0
        for c in clusters[:args.top_n]:
            ov = residue_overlap(c.lining_residues, known)
            if ov > best_ov:
                best_ov, best_rank = ov, c.rank
                best_ov_size = len(_found_resnums(c.lining_residues))
            jac = residue_jaccard(c.lining_residues, known)
            if jac > best_jac:
                best_jac, best_jac_rank = jac, c.rank
            best_core = max(best_core, hotspot_core_overlap(c.centroid, known_ca))

        # Top-k detection curve: size-robust headline hit within the top-k clusters,
        # for a range of k. A flat curve means ranking is fine and the ceiling is
        # detection/sampling; a rising curve means the pocket is found but mis-ranked.
        topk_hit: dict[int, bool] = {}
        for k in (1, 3, 5, 10, 20):
            jk = max((residue_jaccard(c.lining_residues, known) for c in clusters[:k]), default=0.0)
            dk, _ = pocket_min_centroid_dist(clusters, ref_centroid, top_n=k)
            topk_hit[k] = (dk <= CENTROID_THRESHOLD) or (jk >= JACCARD_THRESHOLD)

        dist_ok = best_dist <= CENTROID_THRESHOLD
        ov_ok = best_ov >= OVERLAP_THRESHOLD
        jac_ok = best_jac >= JACCARD_THRESHOLD
        core_ok = best_core >= HOTSPOT_CORE_THRESHOLD

        # Legacy criterion (kept for backward comparison): centroid OR recall.
        found = dist_ok or ov_ok
        status = "PASS" if found else "MISS"
        # Size-robust criterion: centroid proximity OR Jaccard ≥ threshold. This is
        # the honest headline - it cannot be gamed by returning larger pockets.
        found_robust = dist_ok or jac_ok
        status_robust = "PASS" if found_robust else "MISS"

        marker = "✅" if found_robust else "❌"
        dist_str = f"{best_dist:.1f}Å" if best_dist < float("inf") else "n/a"
        pass_by = ("dist" if dist_ok else "") + ("+" if (dist_ok and jac_ok) else "") + ("jac" if jac_ok else "")
        print(f"  {marker} robust:{status_robust}({pass_by or 'none'})  legacy:{status}  "
              f"dist={dist_str}@r{best_dist_rank}  jac={best_jac:.0%}@r{best_jac_rank}  "
              f"core={best_core:.0%}  recall={best_ov:.0%}@r{best_rank}(n={best_ov_size})  "
              f"clusters={len(clusters)}  {elapsed:.1f}s")

        results.append({
            "id": entry["id"],
            "name": entry["name"],
            "category": entry["category"],
            "mechanism": MECHANISM.get(entry["id"], "unknown"),
            "apo_pdb": entry["apo_pdb"],
            "status": status_robust,
            "status_legacy": status,
            "pass_by": pass_by if found_robust else None,
            "centroid_dist_A": round(best_dist, 2) if best_dist < float("inf") else None,
            "centroid_rank": best_dist_rank,
            "overlap": round(best_ov, 3),
            "overlap_found_size": best_ov_size,
            "jaccard": round(best_jac, 3),
            "jaccard_rank": best_jac_rank,
            "hotspot_core": round(best_core, 3),
            "topk_hit": topk_hit,
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
            d = r.get("centroid_dist_A")
            dist_s = f"{d:.1f}Å" if d is not None else "n/a "
            jac = f"{r['jaccard']:.0%}" if r.get("jaccard") is not None else "-"
            ov = f"{r['overlap']:.0%}" if r.get("overlap") is not None else "-"
            t = f"{r['elapsed_s']:.1f}s" if r.get("elapsed_s") is not None else ""
            print(f"    {m} {r['apo_pdb']}  {dist_s:6s} jac={jac:4s} recall={ov:5s}  {t:6s}  {r['name']}")
        grand_pass += n_pass
        grand_total += len(cat_rows)

    cryptic_rows = [r for r in ran if r.get("category") == "cryptic"]
    cr_pass = sum(1 for r in cryptic_rows if r["status"] == "PASS")
    cr_pass_legacy = sum(1 for r in cryptic_rows if r.get("status_legacy") == "PASS")

    # Per-metric transparency. The headline is the size-robust "either" (centroid OR
    # Jaccard). We also report the legacy recall criterion and a Jaccard threshold
    # sweep so the reader can see exactly how much the old number was inflated by
    # size-gaming and how sensitive the honest number is to the Jaccard cutoff.
    def _metric_counts(rows):
        dist = sum(1 for r in rows if (r.get("centroid_dist_A") or 1e9) <= CENTROID_THRESHOLD)
        recall = sum(1 for r in rows if (r.get("overlap") or 0.0) >= OVERLAP_THRESHOLD)
        jac = {t: sum(1 for r in rows if (r.get("jaccard") or 0.0) >= t)
               for t in (0.20, 0.25, 0.30)}
        core = sum(1 for r in rows if (r.get("hotspot_core") or 0.0) >= HOTSPOT_CORE_THRESHOLD)
        robust = sum(1 for r in rows if r["status"] == "PASS")           # dist OR jac≥0.25
        legacy = sum(1 for r in rows if r.get("status_legacy") == "PASS")  # dist OR recall
        return dist, recall, jac, core, robust, legacy

    print(f"\n{'─'*70}")
    print(f"  CRYPTIC ONLY - size-robust headline: {cr_pass}/{len(cryptic_rows)}"
          f"   (legacy recall-based: {cr_pass_legacy}/{len(cryptic_rows)})")
    print(f"  TOTAL (all categories, size-robust): {grand_pass}/{grand_total}")
    print(f"  Size-robust criterion (top-{args.top_n}, OR): centroid ≤ {CENTROID_THRESHOLD} Å "
          f"OR Jaccard ≥ {JACCARD_THRESHOLD:.0%}")
    print(f"  Legacy criterion (size-gameable): centroid ≤ {CENTROID_THRESHOLD} Å "
          f"OR recall ≥ {OVERLAP_THRESHOLD:.0%}")
    print(f"\n  Per-metric breakdown (transparency):")
    hdr = (f"    {'category':16s}  cen≤{CENTROID_THRESHOLD:.0f}Å  "
           f"jac≥.20  jac≥.25  jac≥.30  core≥{HOTSPOT_CORE_THRESHOLD:.0%}  "
           f"recall≥{OVERLAP_THRESHOLD:.0%}  ROBUST  legacy")
    print(hdr)
    for cat in categories + ["__all__"]:
        rows = ran if cat == "__all__" else [r for r in ran if r.get("category") == cat]
        if not rows:
            continue
        dist, recall, jac, core, robust, legacy = _metric_counts(rows)
        label = "ALL" if cat == "__all__" else cat
        print(f"    {label:16s}  {dist:5d}  {jac[0.20]:6d}  {jac[0.25]:6d}  "
              f"{jac[0.30]:6d}  {core:7d}  {recall:8d}  {robust:5d}  {legacy:5d}   (n={len(rows)})")

    # Per-mechanism stratification (cryptic only): which OPENING MECHANISM class
    # does the sampler handle, and which does it fail? Small N per class - a
    # diagnostic of where sampling breaks down (e.g. interface vs side-chain), not
    # a statistical claim. Mechanism labels are the coarse literature annotations
    # in MECHANISM above.
    mech_order = ["sidechain", "loop", "helix", "hinge", "interface"]
    print(f"\n  Cryptic by opening mechanism (size-robust ROBUST / n):")
    for mech in mech_order:
        mrows = [r for r in cryptic_rows if r.get("mechanism") == mech]
        if not mrows:
            continue
        npass = sum(1 for r in mrows if r["status"] == "PASS")
        ids = ", ".join(r["apo_pdb"] for r in mrows if r["status"] != "PASS")
        miss_s = f"  misses: {ids}" if ids else ""
        print(f"    {mech:10s}  {npass}/{len(mrows)}{miss_s}")

    # Top-k detection curve + bootstrap CI on the headline (top-5). A flat curve
    # (top-5 ≈ top-20) means the ceiling is detection/sampling, not ranking. The CI
    # is over targets (resampled), the unit an honest claim is made over - reporting
    # a range instead of a single number is the guard against over-claiming.
    try:
        from metrics import paired_bootstrap_ci
    except ImportError:
        paired_bootstrap_ci = None
    ks = (1, 3, 5, 10, 20)
    with_curve = [r for r in ran if isinstance(r.get("topk_hit"), dict)]
    if with_curve:
        print(f"\n  Size-robust top-k detection curve (all {len(with_curve)} run):")
        for k in ks:
            hits = [bool(r["topk_hit"].get(str(k), r["topk_hit"].get(k))) for r in with_curve]
            rate = sum(hits) / len(hits)
            ci = ""
            if paired_bootstrap_ci is not None:
                _, lo, hi = paired_bootstrap_ci(hits)
                ci = f"   95% CI [{lo:.0%}, {hi:.0%}]"
            print(f"    top-{k:<2d}: {sum(hits):2d}/{len(hits)} ({rate:.0%}){ci}")

    if skipped:
        print(f"  Skipped: {len(skipped)} "
              f"({', '.join(s['id'] for s in skipped)})")

    # Save JSON
    out_path = Path(__file__).parent / "cryptic_benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Full results → {out_path}")
    print(f"\n  Tool citation: {TOOL_CITATION}")


if __name__ == "__main__":
    main()
