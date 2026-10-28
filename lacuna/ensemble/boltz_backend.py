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

from lacuna.ensemble.base import EnsembleBackend

#: Below this fraction of input atoms matched, a predicted frame is discarded
#: rather than used. Matches the external backend's threshold.
_MIN_MATCH_FRACTION = 0.5


def _superpose(mobile: np.ndarray, target: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Kabsch-superpose `mobile` onto `target` using only the atoms in `idx`.

    Returns the full `mobile` array rotated and translated into the target frame.
    Unmatched atoms carry the reference coordinate already, so they are invariant
    under the transform by construction and need no special handling.
    """
    if idx.size < 3:
        return mobile
    m, t = mobile[idx], target[idx]
    mc, tc = m.mean(axis=0), t.mean(axis=0)
    cov = (m - mc).T @ (t - tc)
    u, _s, vt = np.linalg.svd(cov)
    # Reflection guard: without it an improper rotation can mirror the structure.
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return (mobile - mc) @ rot.T + tc


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
            keys = [(a.chain_id, a.res_seq, a.name) for a in input_structure.atoms]

            for pdb_file in pdb_files:
                try:
                    s = load_structure(pdb_file)
                except Exception:
                    continue
                frame = {(a.chain_id, a.res_seq, a.name): c
                         for a, c in zip(s.atoms, coords_array(s))}
                # Boltz predicts from sequence alone, so it may reorder atoms and
                # will in general return a different count. Match on identity and
                # keep the reference coordinate where an atom is missing; the
                # previous positional take-the-first-N with zero padding silently
                # placed unmatched atoms at the origin.
                out = base.copy()
                matched_idx = []
                for i, key in enumerate(keys):
                    got = frame.get(key)
                    if got is not None:
                        out[i] = got
                        matched_idx.append(i)
                frac = len(matched_idx) / max(len(keys), 1)
                if frac < _MIN_MATCH_FRACTION:
                    warnings.warn(
                        f"{pdb_file.name} matched only {frac:.0%} of input atoms; "
                        "skipping this conformer",
                        RuntimeWarning, stacklevel=2)
                    continue
                # Boltz folds de novo in its own frame. Cluster identity downstream
                # is absolute-Cartesian centroid proximity, so an unaligned frame
                # would put every pocket in its own cluster and report the whole
                # protein as cryptic. Superpose on the matched atoms.
                idx = np.asarray(matched_idx)
                out = _superpose(out, base, idx)
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
