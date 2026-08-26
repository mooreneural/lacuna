"""Run IF-SitePred and return ranked residue sets.

IF-SitePred needs two mutually incompatible environments, which is why it is not
simply a function in collect_candidates.py. Its ESM-IF1 stack pins numpy below 2
(via biotite < 1.0, which still exports filter_backbone) while its clustering
step wants PyMOL. So residue prediction runs in one interpreter and centre
clustering in another, exactly as its README instructs.

Two adaptations, both documented in the manuscript and both applied here rather
than by editing the vendored clone, so the third-party source stays pristine and
the changes stay visible in our code:

1. The released pipeline reports at most three sites per structure. That would
   make coverage and top-3 identical by construction, which is precisely the
   quantity this benchmark measures, so the cap is lifted and every cluster is
   kept. Its own ranking, by DBSCAN cluster size, is untouched.

2. Its sites are point clouds, not residue sets. A site's residues are those
   with any atom within 4.5 A of its points, the radius the method's own cloud
   construction uses to associate probe points with a residue.

Nothing else is changed: the LightGBM ensemble, the embeddings, the clustering
parameters and the ranking are as released.
"""
from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

IFSP_DIR = Path("/root/ifsitepred")
IFSP_PY = Path("/root/ifsitepred_venv/bin/python")
PYMOL_PY = Path("/root/lacuna_venv/bin/python")

#: The radius the method's own cloud construction uses.
CLOUD_RADIUS = 4.5

#: The released cap, and what it becomes. Patched into a copy at run time.
_CAP_SRC = "range(min([3,len(site_counts)]))"
_CAP_DST = "range(len(site_counts))"


class IFSitePredUnavailable(RuntimeError):
    pass


def _check_install():
    missing = [str(p) for p in (IFSP_DIR, IFSP_PY, PYMOL_PY) if not p.exists()]
    if missing:
        raise IFSitePredUnavailable(
            "IF-SitePred is not installed here: missing %s. Clone "
            "https://github.com/annacarbery/binding-sites to %s and build its "
            "environment." % (", ".join(missing), IFSP_DIR))
    if not (IFSP_DIR / "models" / "lgbm_if1_0.pkl").exists():
        raise IFSitePredUnavailable(
            "IF-SitePred clone at %s has no models/; the checkout is incomplete."
            % IFSP_DIR)


def _patched_centres_script() -> Path:
    """A copy of predict_centres.py with the three-site cap lifted.

    Written beside the original so its relative paths still resolve, and
    regenerated each run so it can never drift from the vendored source.
    """
    src = IFSP_DIR / "src" / "predict_centres.py"
    text = src.read_text(encoding="utf-8")
    normalised = re.sub(r"range\(min\(\[\s*3\s*,\s*len\(site_counts\)\s*\]\)\)",
                        _CAP_DST, text)
    if normalised == text and _CAP_DST not in text:
        raise IFSitePredUnavailable(
            "could not find the three-site cap in %s. IF-SitePred's source has "
            "changed; re-check the adaptation before trusting these numbers." % src)
    out = IFSP_DIR / "src" / "_predict_centres_uncapped.py"
    out.write_text(normalised, encoding="utf-8")
    return out


def _read_cloud(path: Path):
    """Points from one of IF-SitePred's .xyz files.

    Format is an atom count, a comment line, then 'label x y z' per point.
    """
    pts = []
    for line in path.read_text(encoding="utf-8").splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 4:
            try:
                pts.append([float(v) for v in parts[-3:]])
            except ValueError:
                continue
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 3))


def _residues_near(structure, cloud, chain, radius=CLOUD_RADIUS):
    """Residue labels with any atom within `radius` of the cloud."""
    if not len(cloud):
        return []
    xyz, resnum = [], []
    for a in structure.atoms:
        xyz.append(a.coords)
        resnum.append(a.res_seq)
    xyz = np.asarray(xyz, dtype=float)
    resnum = np.asarray(resnum)
    d = np.linalg.norm(xyz[:, None, :] - cloud[None, :, :], axis=-1).min(axis=1)
    hits = sorted({int(n) for n in resnum[d <= radius]})
    return ["%d:%s" % (n, chain) for n in hits]


def run_ifsitepred(pdb_path: Path, chain: str, structure, target: str,
                   timeout: int = 3600):
    """Ranked residue-label lists for one structure, best first."""
    _check_install()
    # The scripts os.mkdir into these without creating the parents, and git does
    # not track empty directories, so a fresh clone has neither.
    for d in ("input", "predictions", "xyz"):
        (IFSP_DIR / d).mkdir(exist_ok=True)
    inp = IFSP_DIR / "input"
    shutil.copy(pdb_path, inp / ("%s.pdb" % target))
    pred_dir = IFSP_DIR / "predictions" / target
    if pred_dir.exists():
        shutil.rmtree(pred_dir)

    r = subprocess.run(
        [str(IFSP_PY), "src/predict_residues.py", "-t", target],
        cwd=IFSP_DIR, capture_output=True, text=True, timeout=timeout)
    if not (pred_dir / "predicted_residues.json").exists():
        raise RuntimeError("predict_residues failed: %s" % r.stderr[-300:])

    centres = _patched_centres_script()
    r = subprocess.run(
        [str(PYMOL_PY), str(centres.relative_to(IFSP_DIR)), "-t", target],
        cwd=IFSP_DIR, capture_output=True, text=True, timeout=timeout)

    clouds = sorted(pred_dir.glob("site_rank_*.xyz"),
                    key=lambda p: int(re.search(r"_(\d+)\.xyz$", p.name).group(1)))
    if not clouds:
        # Finding nothing is a legitimate prediction, not a failure, and must be
        # recorded as an empty candidate list. An earlier version only accepted
        # that when the child happened to print an exact phrase, so structures
        # where IF-SitePred genuinely found no site were dropped as errors, which
        # silently removed its hardest cases and inflated its coverage. The
        # process exit status is the thing that distinguishes the two.
        if r.returncode == 0:
            return []
        raise RuntimeError("predict_centres exited %d: %s"
                           % (r.returncode, (r.stderr or "")[-300:]))

    return [_residues_near(structure, _read_cloud(p), chain) for p in clouds]
