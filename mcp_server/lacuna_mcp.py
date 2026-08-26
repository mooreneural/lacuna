#!/usr/bin/env python
"""MCP server exposing Lacuna's cryptic-pocket discovery to Claude.

Runs locally over stdio: Claude launches this as a subprocess, it calls Lacuna
on the user's own machine, and nothing leaves it. No hosting, no API keys, no
network round trip. Lacuna's default backend is CPU normal mode analysis and
finishes in seconds, so this is a reasonable thing to run interactively.

Install:

    pip install lacuna-pockets mcp

Register with Claude Code, from the directory containing this file:

    claude mcp add lacuna -- python /absolute/path/to/lacuna_mcp.py

Or add to Claude Desktop's claude_desktop_config.json:

    {
      "mcpServers": {
        "lacuna": {
          "command": "python",
          "args": ["/absolute/path/to/lacuna_mcp.py"]
        }
      }
    }

Design notes:

Tools shell out to the installed `lacuna` CLI rather than importing the pipeline.
That keeps this server working against whatever version the user has, isolates a
crash in the science code from the transport, and means the tested code path is
the one that runs. The cost is a subprocess per call, which is noise next to the
ensemble generation.

Structures may be given as a local path or a four-character PDB id. The id form
downloads from RCSB into a cache directory, because the alternative is the user
hunting for a file path before they can ask a question.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from mcp.server import MCPServer

#: FastMCP in MCP SDK 1.x; renamed MCPServer in 2.0. Same decorator API.
mcp = MCPServer(
    "lacuna",
    instructions=(
        "Finds cryptic binding pockets: sites closed or absent in a protein's "
        "unbound structure that open only as it moves. Runs locally on CPU in "
        "seconds. Use for cryptic, transient, hidden or allosteric pockets, not "
        "for sites already open in the structure."
    ),
)

CACHE = Path(os.environ.get("LACUNA_MCP_CACHE",
                            Path(tempfile.gettempdir()) / "lacuna_mcp"))
TIMEOUT = int(os.environ.get("LACUNA_MCP_TIMEOUT", "900"))


def _exe() -> list[str]:
    """The lacuna entry point, however it happens to be installed."""
    found = shutil.which("lacuna")
    if found:
        return [found]
    # Installed as a library but not on PATH: click's group is importable.
    return [sys.executable, "-c",
            "from lacuna.cli import main; main()"]


def _resolve(structure: str) -> Path:
    """A local path, or a PDB id fetched into the cache."""
    p = Path(structure).expanduser()
    if p.exists():
        return p
    ident = structure.strip().lower()
    if len(ident) == 4 and ident.isalnum():
        CACHE.mkdir(parents=True, exist_ok=True)
        dst = CACHE / ("%s.cif" % ident)
        if not dst.exists():
            urllib.request.urlretrieve(
                "https://files.rcsb.org/download/%s.cif" % ident.upper(), dst)
        return dst
    raise FileNotFoundError(
        "No file at %r, and %r is not a 4-character PDB id. Give an absolute "
        "path to a .pdb or .cif file, or a PDB id such as 1A4U." % (structure, structure))


def _run(args: list[str], out_dir: Path) -> dict:
    cmd = _exe() + args
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT)
    except FileNotFoundError:
        raise RuntimeError(
            "Lacuna is not installed in this environment. Install it with "
            "`pip install lacuna-pockets`.")
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            "Lacuna exceeded %d s. Large proteins with many conformers can be "
            "slow; try fewer conformers, or raise LACUNA_MCP_TIMEOUT." % TIMEOUT)
    report = out_dir / "pocket_report.json"
    if not report.exists():
        tail = (r.stderr or r.stdout or "").strip().splitlines()[-12:]
        raise RuntimeError("Lacuna produced no report (exit %d).\n%s"
                           % (r.returncode, "\n".join(tail)))
    return json.loads(report.read_text())


@mcp.tool()
def find_cryptic_pockets(
    structure: str,
    conformers: int = 20,
    top: int = 10,
    min_crypticity: float = 0.0,
    min_druggability: float = 0.0,
    rank_by: str = "learned",
) -> str:
    """Find cryptic binding pockets in a protein structure.

    Cryptic pockets are binding sites that are closed, too small, or absent in a
    protein's unbound (apo) structure and open only as the protein moves. Ordinary
    pocket finders score a single static structure and therefore cannot see them.
    Lacuna generates a conformational ensemble, detects pockets in every conformer,
    groups detections that correspond to the same physical site, and ranks the
    sites by geometry and by how they behave across the ensemble.

    Use this when someone asks about cryptic, transient, hidden, or allosteric
    pockets, about druggability of a target with no known pocket, or about
    binding sites that only appear on motion. For pockets already open in the
    structure, a single-structure detector is a better fit.

    Returns ranked sites as JSON. Each carries a centroid, lining residues,
    volume across the ensemble, persistence (fraction of conformers the site
    appears in), crypticity (how much it opens relative to the starting
    structure), and a druggability score.

    Args:
        structure: Path to a .pdb or .cif file, or a 4-character PDB id (e.g. "1A4U").
        conformers: Ensemble size, 20 by default. Larger ensembles find more sites
            but return more candidates, which pushes the true site further down the
            ranking; the default balances the two.
        top: Maximum sites to return.
        min_crypticity: Keep only sites opening at least this much, 0 to 1. Use ~0.3
            to restrict to genuinely cryptic sites rather than already-open ones.
        min_druggability: Keep only sites scoring at least this, 0 to 1.
        rank_by: Ranking strategy. "learned" is the shipped default; "learned-plm"
            adds protein-language-model features and ranks better, at the cost of
            downloading a 2.5 GB model on first use.
    """
    path = _resolve(structure)
    out = Path(tempfile.mkdtemp(prefix="lacuna_"))
    args = ["discover", str(path), "--output", str(out), "--quiet",
            "--conformers", str(int(conformers)), "--top", str(int(top)),
            "--min-crypticity", str(float(min_crypticity)),
            "--min-druggability", str(float(min_druggability))]
    if rank_by.strip():
        args += ["--rank-by", rank_by.strip()]
    report = _run(args, out)

    # Field names come from PocketCluster.to_dict(). Volume keys carry a capital
    # A ("volume_A3"); guessing the lowercase form silently emits null.
    pockets = []
    for p in report.get("pockets", []):
        pockets.append({
            "rank": p.get("rank"),
            "centroid": p.get("centroid"),
            "volume_A3": p.get("volume_A3"),
            "volume_range_A3": p.get("volume_range_A3"),
            "apo_volume_A3": p.get("apo_volume_A3"),
            "druggability": p.get("druggability"),
            "max_druggability": p.get("max_druggability"),
            "crypticity": p.get("crypticity"),
            "cryptic": p.get("cryptic"),
            "persistence": p.get("persistence"),
            "n_conformers_present": len(p.get("appears_in_conformers") or []),
            "lining_residues": p.get("lining_residues"),
        })
    return json.dumps({
        "protein": report.get("protein"),
        "resolved_from": str(path),
        "n_conformers": report.get("n_conformers"),
        "n_pockets_found": report.get("n_pockets_found"),
        "n_cryptic_pockets": report.get("n_cryptic_pockets"),
        "ranked_by": report.get("ranked_by"),
        "pockets": pockets,
        "note": ("Ranked sites are candidates for follow-up, not validated binding "
                 "sites. Crypticity above ~0.3 indicates a site that opens "
                 "substantially relative to the starting structure."),
    }, indent=2)


@mcp.tool()
def export_docking_region(
    structure: str,
    conformers: int = 20,
    top: int = 3,
    fmt: str = "vina",
) -> str:
    """Emit docking search regions for a protein's top cryptic pockets.

    Runs the same discovery as find_cryptic_pockets, then writes inputs that
    downstream docking and co-folding tools consume directly: AutoDock Vina and
    GNINA search boxes, or Boltz constraint files.

    Use this when someone wants to dock or co-fold against a cryptic pocket and
    needs the search region, rather than just a list of sites.

    Args:
        structure: Path to a .pdb or .cif file, or a 4-character PDB id.
        conformers: Ensemble size; 20 is the validated default.
        top: How many of the highest-ranked pockets to emit regions for.
        fmt: "vina" for AutoDock Vina and GNINA boxes, "boltz" for Boltz constraints.
    """
    path = _resolve(structure)
    out = Path(tempfile.mkdtemp(prefix="lacuna_"))
    flag = "--emit-vina-boxes" if fmt.lower().startswith("v") else "--emit-boltz-constraints"
    report = _run(["discover", str(path), "--output", str(out), "--quiet",
                   "--conformers", str(int(conformers)), "--top", str(int(top)),
                   flag], out)

    files = []
    for f in sorted(out.iterdir()):
        if f.is_file() and f.name != "pocket_report.json":
            try:
                files.append({"file": f.name, "content": f.read_text()[:4000]})
            except UnicodeDecodeError:
                files.append({"file": f.name, "content": "<binary>"})
    return json.dumps({
        "protein": report.get("protein"),
        "format": "vina" if flag.endswith("vina-boxes") else "boltz",
        "output_dir": str(out),
        "n_pockets": report.get("n_pockets_found"),
        "files": files,
        "note": ("These are search regions, not receptors. A cryptic site is "
                 "closed in the deposited structure and only partly opened in a "
                 "generated conformer, so neither file is a reliable rigid "
                 "receptor. Use the region to focus flexible-receptor docking, "
                 "or to place a co-folding constraint, rather than docking "
                 "rigidly against either structure."),
    }, indent=2)


if __name__ == "__main__":
    mcp.run()
