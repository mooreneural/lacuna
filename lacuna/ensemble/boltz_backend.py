"""Boltz-2 partial diffusion ensemble backend.

Uses Boltz-2's learned diffusion prior to generate physically realistic
alternate conformations by partially noising the input structure and
re-denoising from an intermediate sigma level.

Partial diffusion gives better conformational diversity than running from
pure noise (which would ignore the input structure entirely) while staying
within the learned energy landscape.

Requires: pip install lacuna[boltz]  (+ GPU strongly recommended)

NOTE: Requires Boltz >= 0.4 which must expose `partial_diffusion` in its
      Python API. If that API is not yet available, this backend falls back
      to running multiple full-noise predictions (still better than random).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from lacuna.ensemble.base import EnsembleBackend


class BoltzBackend(EnsembleBackend):
    """Partial diffusion via Boltz-2 for highest-quality conformational ensembles."""

    def __init__(
        self,
        noise_fraction: float = 0.4,
        sampling_steps: int = 100,
        cache_dir: str | None = None,
    ):
        # noise_fraction: fraction of sigma_max to use as starting noise level
        # 0.2 = local flexibility, 0.5 = domain rearrangements, 0.8 = global resampling
        self.noise_fraction = noise_fraction
        self.sampling_steps = sampling_steps
        self.cache_dir = cache_dir

    @property
    def name(self) -> str:
        return "boltz"

    def generate(
        self,
        structure_path: Path,
        n_conformers: int,
        **kwargs,
    ) -> list[np.ndarray]:
        try:
            import boltz  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Boltz backend requires boltz to be installed. "
                "Install with: pip install lacuna[boltz]"
            ) from e

        # Try partial diffusion API (requires boltz >= 0.4 with Lacuna patch)
        try:
            return self._partial_diffusion(structure_path, n_conformers)
        except (AttributeError, ImportError):
            return self._full_diffusion_fallback(structure_path, n_conformers)

    def _partial_diffusion(self, structure_path: Path, n_conformers: int) -> list[np.ndarray]:
        """Use partial diffusion if the Boltz API supports initial_coords."""
        from boltz.model.models.boltz2 import Boltz2
        from boltz.data.parse.mmcif import parse_mmcif
        from boltz.data.parse.pdb import parse_pdb

        suffix = structure_path.suffix.lower()
        if suffix in (".cif", ".mmcif"):
            structure = parse_mmcif(structure_path)
        else:
            structure = parse_pdb(structure_path)

        model = Boltz2.load_from_checkpoint(
            self._get_checkpoint(),
            map_location="cpu",
        )
        model.eval()

        conformers: list[np.ndarray] = []
        noise_levels = np.linspace(
            self.noise_fraction * 0.5,
            self.noise_fraction * 1.5,
            n_conformers,
        ).clip(0.05, 0.95)

        for noise_frac in noise_levels:
            coords = model.partial_diffusion(
                structure=structure,
                noise_fraction=float(noise_frac),
                sampling_steps=self.sampling_steps,
            )
            conformers.append(coords.numpy().astype(np.float32))

        return conformers

    def _full_diffusion_fallback(self, structure_path: Path, n_conformers: int) -> list[np.ndarray]:
        """Fallback: run boltz predict CLI N times with different seeds."""
        import subprocess
        import json

        conformers: list[np.ndarray] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(n_conformers):
                out_dir = Path(tmpdir) / f"sample_{i}"
                cmd = [
                    "boltz", "predict", str(structure_path),
                    "--output", str(out_dir),
                    "--diffusion_samples", "1",
                    "--model", "boltz2",
                    "--step_scale", "1.8",  # higher temperature for diversity
                ]
                if self.cache_dir:
                    cmd += ["--cache", self.cache_dir]

                subprocess.run(cmd, check=True, capture_output=True)

                # Parse output coords from the CIF file
                cif_files = list(out_dir.glob("predictions/**/*_model_0.cif"))
                if cif_files:
                    from lacuna.io.structure import load_structure, coords_array
                    s = load_structure(cif_files[0])
                    conformers.append(coords_array(s))

        return conformers

    def _get_checkpoint(self) -> str:
        from pathlib import Path as P
        import os
        cache = P(self.cache_dir or os.path.expanduser("~/.boltz"))
        ckpts = list(cache.glob("boltz2*.ckpt"))
        if not ckpts:
            raise FileNotFoundError(
                "No Boltz-2 checkpoint found. Run `boltz predict` once to download weights."
            )
        return str(sorted(ckpts)[-1])
