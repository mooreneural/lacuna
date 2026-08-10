# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Clayton Moore
"""Protein language model signal for pocket ranking (optional).

Why sequence, when Lacuna is a geometry tool
--------------------------------------------
Ranking is where recovery is lost: the detector proposes a well-localized pocket
for roughly three quarters of CryptoBench structures, but only some of that
reaches the top five. Adding more *geometric* opinion does not fix it. Feeding
the ranker P2Rank's own per-pocket confidence, a second geometric view of the
same cavity, measured -1.1% on held-out data.

Sequence is a different axis. A protein language model has learned which
positions are evolutionarily constrained, which correlates with ligand binding
and is invisible to a distance transform. On CryptoBench's held-out test fold
this takes top-5 recovery from 57.0% to 65.4% (paired +8.4%, 95% CI +4.5 to
+12.8), closing the gap to P2Rank.

How it works
------------
1. ESM-2 embeds the sequence once per structure. The sequence does not change as
   the geometry does, so one forward pass is reused across every conformer: this
   costs 1x, not Nx.
2. A linear head turns each residue embedding into P(residue lines a cryptic
   site). Held-out per-residue AUC 0.855, average precision 0.292 against a 0.049
   baseline.
3. Each pocket is summarized by four statistics of that probability over its own
   lining residues (see ``pocket_features``). Four scalars rather than the raw
   1280 dimensions: appending the embedding itself to a 23-feature ranker would
   overfit long before it generalized.

Optional by design
------------------
This needs ``torch`` and ``transformers`` and downloads a model on first use,
which is at odds with the CPU-only, NumPy-and-SciPy-only default. It therefore
ships as an extra:

    pip install "lacuna-pockets[plm]"

Without it, ``available()`` is False and Lacuna ranks on geometry alone. Nothing
raises unless the sequence ranker is explicitly requested.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np

from lacuna.io.structure import _THREE_TO_ONE

#: ESM-2 checkpoint the shipped head was fitted against. The head weights are
#: tied to this model's embedding space; changing it requires refitting.
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"

#: Feature names contributed to the ranker, in the order the head produces them.
FEATURES = ("plm_mean", "plm_max", "plm_top3", "plm_frac")

_HEAD_PATH = Path(__file__).with_name("plm_head.npz")


def available() -> bool:
    """True if the optional dependencies and the fitted head are both present."""
    if not _HEAD_PATH.exists():
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        return False
    return True


def _require() -> None:
    if not _HEAD_PATH.exists():
        raise RuntimeError(
            f"sequence head not found at {_HEAD_PATH}; reinstall lacuna-pockets "
            "or refit with benchmarks/train_ranker.py"
        )
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            'the sequence ranker needs the "plm" extra: '
            'pip install "lacuna-pockets[plm]"'
        ) from exc


@functools.lru_cache(maxsize=1)
def _load_model():
    """Load the embedder once per process; it is several GB and slow to build."""
    _require()
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).eval()
    cuda = torch.cuda.is_available()
    if cuda:
        # Half precision on GPU: the head is a linear probe, and the accuracy
        # difference is far below the noise floor of the benchmark.
        model = model.cuda().half()
    head = np.load(_HEAD_PATH)
    return tok, model, cuda, head["w"].astype(np.float32), float(head["b"])


def sequence_of(structure) -> tuple[str, list[int]]:
    """One-letter sequence and the residue numbers it corresponds to, in order.

    Unknown residue names become ``X``, which ESM-2 handles natively, so a
    modified or non-standard residue costs one uninformative position rather
    than shifting every downstream index.
    """
    seq, nums = [], []
    for r in structure.residues:
        seq.append(_THREE_TO_ONE.get(r.name, "X"))
        nums.append(r.seq_num)
    return "".join(seq), nums


def residue_probabilities(structure) -> dict[int, float]:
    """Map residue number -> P(residue lines a cryptic site).

    Computed once per structure. Callers running an ensemble should compute this
    on the input structure and reuse it for every conformer.
    """
    import torch

    tok, model, cuda, w, b = _load_model()
    seq, nums = sequence_of(structure)
    if not seq:
        return {}
    with torch.no_grad():
        enc = tok(seq, return_tensors="pt")
        if cuda:
            enc = {k: v.cuda() for k, v in enc.items()}
        # Strip the BOS/EOS tokens so positions line up with residues.
        emb = model(**enc).last_hidden_state[0, 1:-1].float().cpu().numpy()
    if len(emb) != len(nums):
        # Tokenizer disagreed with the residue list; better to contribute nothing
        # than to silently misalign sequence signal with the wrong residues.
        return {}
    logits = emb @ w + b
    return dict(zip(nums, (1.0 / (1.0 + np.exp(-logits))).tolist()))


def pocket_features(lining_resnums, prob_by_residue: dict[int, float]) -> dict[str, float]:
    """Summarize a pocket by the sequence bindingness of the residues lining it.

    ``mean`` is the strongest single term; ``max`` saturates near 1.0 for most
    pockets and mainly serves to temper it; ``top3`` is a middle ground robust to
    one lucky residue; ``frac`` captures how much of the pocket looks binding-like
    rather than how much its best residue does.
    """
    vals = sorted((prob_by_residue[r] for r in lining_resnums if r in prob_by_residue),
                  reverse=True)
    if not vals:
        return dict.fromkeys(FEATURES, 0.0)
    return {
        "plm_mean": float(np.mean(vals)),
        "plm_max": float(vals[0]),
        "plm_top3": float(np.mean(vals[:3])),
        "plm_frac": float(np.mean([v >= 0.5 for v in vals])),
    }
