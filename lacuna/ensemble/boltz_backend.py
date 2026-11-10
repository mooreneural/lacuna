# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Boltz-2 ensemble backend.

Generates a conformational ensemble by running a single `boltz predict` call
with --diffusion_samples N.  Each diffusion sample is an independent draw from
the learned posterior over structures, giving physically realistic diversity.

step_scale < default (1.5) increases diversity; 1.2-1.3 is good for cryptic
pocket sampling.  step_scale > default gives more conservative sampling.

Requires: pip install "lacuna-pockets[boltz]"  (+ GPU strongly recommended)
"""

from __future__ import annotations

import subprocess
import tempfile
import warnings
from pathlib import Path

import numpy as np

from lacuna.ensemble.base import MIN_MATCH_FRACTION, EnsembleBackend, align_frame

#: Below this fraction of input atoms matched, a predicted frame is discarded
#: rather than used. Shared with the other backends.
_MIN_MATCH_FRACTION = MIN_MATCH_FRACTION

#: Below this fraction of PAIRED residues sharing a residue name, a frame is
#: discarded. Matching on atom name alone cannot detect a numbering offset,
#: because every residue carries N/CA/C/O: on 1EVY (numbered 9-357) against a
#: Boltz frame numbered 1-346, 70% of atoms matched by name while only 7% of
#: them were the same residue, yielding a 19 A conformer from a fold that was
#: correct to 0.92 A. Residue identity is what actually detects that.
_MIN_IDENTITY_FRACTION = 0.9


class BoltzBackend(EnsembleBackend):
    """Boltz-2 multi-sample ensemble for highest-quality conformational diversity."""

    def __init__(
        self,
        step_scale: float = 1.3,
        sampling_steps: int = 200,
        accelerator: str = "gpu",
        cache_dir: str | None = None,
        use_msa_server: bool = True,
        msa_server_url: str = "https://api.colabfold.com",
    ):
        # step_scale: lower = more diversity (1.2–1.5 recommended)
        # 1.3 is a good default for cryptic pocket sampling
        self.step_scale = step_scale
        self.sampling_steps = sampling_steps
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        # MSA options - use_msa_server=True hits ColabFold for MSA generation;
        # improves ECL/loop region accuracy at the cost of a network round-trip.
        self.use_msa_server = use_msa_server
        self.msa_server_url = msa_server_url

    @property
    def name(self) -> str:
        return "boltz"

    def generate(
        self,
        structure_path: Path,
        n_conformers: int,
        chain: str | None = None,
        **kwargs,
    ) -> list[np.ndarray]:
        try:
            import boltz  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Boltz backend requires boltz to be installed. "
                'Run: pip install "lacuna-pockets[boltz]"'
            ) from e

        return self._run_diffusion_samples(Path(structure_path), n_conformers, chain=chain)

    def _run_diffusion_samples(
        self, structure_path: Path, n_conformers: int, chain: str | None = None
    ) -> list[np.ndarray]:
        """Run boltz predict once with --diffusion_samples N to get all conformers."""
        from lacuna.io.structure import load_structure, coords_array

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)

            # Build Boltz YAML input from the structure's sequence
            yaml_path = self._write_input_yaml(structure_path, tmp, chain=chain)

            out_dir = tmp / "boltz_out"
            cmd = [
                "boltz", "predict", str(yaml_path),
                "--out_dir", str(out_dir),
                "--model", "boltz2",
                "--diffusion_samples", str(n_conformers),
                "--sampling_steps", str(self.sampling_steps),
                "--step_scale", str(self.step_scale),
                "--accelerator", self.accelerator,
                "--output_format", "pdb",
                "--no_kernels",  # use pure-PyTorch path; avoids cuequivariance_ops_torch dep
                "--override",
            ]
            if self.cache_dir:
                cmd += ["--cache", self.cache_dir]
            if self.use_msa_server:
                cmd += ["--use_msa_server", "--msa_server_url", self.msa_server_url]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"boltz predict failed:\n{result.stderr[-2000:]}"
                )

            # Output lands in boltz_results_{stem}/ inside out_dir
            stem = yaml_path.stem
            results_dir = out_dir / f"boltz_results_{stem}"
            pdb_files = sorted(results_dir.rglob("*_model_*.pdb"))
            if not pdb_files:
                pdb_files = sorted(results_dir.rglob("*_model_*.cif"))
            if not pdb_files:
                # fallback: search anywhere under out_dir
                pdb_files = sorted(out_dir.rglob("*_model_*.pdb"))
            if not pdb_files:
                pdb_files = sorted(out_dir.rglob("*_model_*.cif"))

            if not pdb_files:
                raise RuntimeError(
                    f"boltz predict succeeded but produced no structure files in {out_dir}"
                )

            conformers: list[np.ndarray] = []
            input_structure = load_structure(structure_path, chain=chain)
            base = coords_array(input_structure)

            # Residues in observation order, with the atom slots each one owns.
            # Boltz renumbers its output 1..N, so the input's own numbering is
            # NOT a usable key: 80% of CryptoBench structures start somewhere
            # other than 1, and matching on res_seq then pairs atoms across an
            # offset. Order is exact, because Boltz folded the sequence that
            # was extracted from this very structure.
            ref_order: list[tuple[str, int]] = []
            ref_slots: dict[tuple[str, int], dict[str, int]] = {}
            ref_name: dict[tuple[str, int], str] = {}
            for i, a in enumerate(input_structure.atoms):
                k = (a.chain_id, a.res_seq)
                if k not in ref_slots:
                    ref_order.append(k)
                    ref_slots[k] = {}
                    ref_name[k] = a.res_name
                ref_slots[k][a.name] = i

            for pdb_file in pdb_files:
                try:
                    s = load_structure(pdb_file)
                except Exception:
                    continue
                # Same decomposition for the predicted frame.
                frame_order: list[tuple[str, int]] = []
                frame_atoms: dict[tuple[str, int], dict[str, np.ndarray]] = {}
                frame_name: dict[tuple[str, int], str] = {}
                for a, c in zip(s.atoms, coords_array(s)):
                    k = (a.chain_id, a.res_seq)
                    if k not in frame_atoms:
                        frame_order.append(k)
                        frame_atoms[k] = {}
                        frame_name[k] = a.res_name
                    frame_atoms[k][a.name] = c

                # Pair residues by position in sequence, then atoms by name
                # within each pair. Keep the reference coordinate wherever the
                # prediction lacks an atom.
                out = base.copy()
                matched_idx = []
                paired = same_name = 0
                for rk, fk in zip(ref_order, frame_order):
                    paired += 1
                    if ref_name[rk] == frame_name[fk]:
                        same_name += 1
                    got = frame_atoms[fk]
                    for name, slot in ref_slots[rk].items():
                        c = got.get(name)
                        if c is not None:
                            out[slot] = c
                            matched_idx.append(slot)

                frac = len(matched_idx) / max(len(base), 1)
                ident = same_name / max(paired, 1)
                if frac < _MIN_MATCH_FRACTION:
                    warnings.warn(
                        f"{pdb_file.name} matched only {frac:.0%} of input atoms; "
                        "skipping this conformer",
                        RuntimeWarning, stacklevel=2)
                    continue
                if ident < _MIN_IDENTITY_FRACTION:
                    warnings.warn(
                        f"{pdb_file.name}: only {ident:.0%} of paired residues "
                        f"share a residue name ({same_name}/{paired}). The "
                        "prediction does not correspond to the input sequence; "
                        "skipping this conformer",
                        RuntimeWarning, stacklevel=2)
                    continue
                # Boltz folds de novo in its own frame. Cluster identity downstream
                # is absolute-Cartesian centroid proximity, so an unaligned frame
                # would put every pocket in its own cluster and report the whole
                # protein as cryptic. Superpose on the matched atoms, and leave the
                # unmatched ones at their reference coordinates.
                out = align_frame(out, base, matched_idx)
                conformers.append(out.astype(np.float32))

            if not conformers:
                raise RuntimeError(
                    "boltz predict produced structures but none matched the input "
                    "atoms well enough to use")
            return conformers

    def _write_input_yaml(
        self, structure_path: Path, out_dir: Path, chain: str | None = None
    ) -> Path:
        """Create a Boltz YAML from the protein sequence in the structure."""
        from lacuna.io.structure import load_structure

        structure = load_structure(structure_path, chain=chain)
        yaml_path = out_dir / f"{structure_path.stem}.yaml"

        lines = ["sequences:"]
        for chain_id, seq in structure.sequence.items():
            chain_lines = [
                f"  - protein:",
                f"      id: {chain_id}",
                f"      sequence: {seq}",
            ]
            if not self.use_msa_server:
                # PLM-only mode - no MSA server required, fast but ECL loops are noisier
                chain_lines.append(f"      msa: empty")
            # When use_msa_server=True: omit msa field so Boltz fetches from ColabFold
            lines += chain_lines

        yaml_path.write_text("\n".join(lines) + "\n")
        return yaml_path
