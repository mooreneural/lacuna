# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Render paper/software_paper.md to .docx.

Reuses make_docx.py's renderer rather than duplicating it, by rebinding the
module's source and output paths before calling build(). Deliberately never
imports make_docx.main(), and never writes manuscript.docx or
manuscript_twocolumn.docx: the manuscript carries hand formatting that a
regeneration would destroy, so the software note gets its own output path and
nothing else in this directory is at risk from running this script.

    python paper/make_software_docx.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import make_docx  # noqa: E402

SRC = HERE / "software_paper.md"
OUT = HERE / "software_paper.docx"


def main() -> None:
    if not SRC.exists():
        sys.exit(f"missing {SRC}")

    manuscript_before = (HERE / "manuscript.docx")
    stamp = manuscript_before.stat().st_mtime if manuscript_before.exists() else None

    make_docx.SRC = SRC
    make_docx.OUT = OUT
    # Point the journal target at the same file we own, so that even if a future
    # edit to build() writes the two-column variant, it cannot land on the
    # manuscript's own output path.
    make_docx.OUT_JOURNAL = HERE / "software_paper_twocolumn.docx"
    make_docx.build(journal=False)

    if stamp is not None and manuscript_before.stat().st_mtime != stamp:
        sys.exit("ABORT: manuscript.docx was modified; this script must never "
                 "touch it. Investigate before trusting the output.")


if __name__ == "__main__":
    main()
