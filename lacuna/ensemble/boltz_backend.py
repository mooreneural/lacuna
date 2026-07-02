# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Clayton Moore
"""Boltz-2 ensemble backend.

Generates a conformational ensemble by running a single `boltz predict` call
with --diffusion_samples N.  Each diffusion sample is an independent draw from
the learned posterior over structures, giving physically realistic diversity.

step_scale < default (1.5) increases diversity; 1.2-1.3 is good for cryptic
pocket sampling.  step_scale > default gives more conservative sampling.

Requires: pip install lacuna[boltz]  (+ GPU strongly recommended)
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

from lacuna.ensemble.base import EnsembleBackend


class BoltzBackend(EnsembleBackend):
    """Boltz-2 multi-sample ensemble for highest-quality conformational diversity."""

    def __init__(
        self,
        step_scale: float = 1.3,
        sampling_steps: int = 200,
        accelerator: str = "gpu",
        cache_dir: str | None = None,
        use_msa_server: bool = False,
        msa_server_url: str = "https://api.colabfold.com",
    ):
        # step_scale: lower = more diversity (1.2–1.5 recommended)
        # 1.3 is a good default for cryptic pocket sampling
        self.step_scale = step_scale
        self.sampling_steps = sampling_steps
        self.accelerator = accelerator
        self.cache_dir = cache_dir
        # MSA options — use_msa_server=True hits ColabFold for MSA generation;
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
                "Run: pip install lacuna[boltz]"
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
            n_atoms = len(input_structure.atoms)

            for pdb_file in pdb_files:
                try:
                    s = load_structure(pdb_file)
                    coords = coords_array(s)
                    # Align atom count — Boltz may add or reorder atoms
                    if coords.shape[0] == n_atoms:
                        conformers.append(coords)
                    else:
                        # Try to match by truncating/padding — take first N atoms
                        n = min(coords.shape[0], n_atoms)
                        padded = np.zeros((n_atoms, 3), dtype=np.float32)
                        padded[:n] = coords[:n]
                        conformers.append(padded)
                except Exception:
                    continue

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
                # PLM-only mode — no MSA server required, fast but ECL loops are noisier
                chain_lines.append(f"      msa: empty")
            # When use_msa_server=True: omit msa field so Boltz fetches from ColabFold
            lines += chain_lines

        yaml_path.write_text("\n".join(lines) + "\n")
        return yaml_path
