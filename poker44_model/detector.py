"""Poker44 bot detector — uid7 (Ares90125/poker7).

Model: **logistic regression** over winsorized behavioral features, scored by
**within-batch ranking**. This is robust to the benchmark-vs-live distribution
shift: winsorizing bounds out-of-distribution live features so the model can't
saturate, and ranking the batch is exactly what the validator's AP reward
optimizes (it only cares about ordering). Params in model.json (reproducible via
train_model.py). Pure-Python at inference.

`score_batch(chunks)` is the real path (validators send a batch); it returns one
rank-based bot-risk score in [0,1] per chunk (higher = more bot-like).
"""
from __future__ import annotations

import json
import math
import os

from poker44_model.features import extract_group_features

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        with open(os.path.join(os.path.dirname(__file__), "model.json")) as fh:
            _MODEL = json.load(fh)
    return _MODEL


def _raw_decision(m, feats):
    """Raw logit on winsorized, standardized features (unbounded, but ordered)."""
    z = m["intercept"]
    for i, name in enumerate(m["features"]):
        x = feats.get(name, 0.0)
        x = min(max(x, m["winsor_lo"][i]), m["winsor_hi"][i])      # winsorize OOD
        z += m["coef"][i] * ((x - m["mean"][i]) / (m["scale"][i] or 1.0))
    return z


def _rank_normalize(vals):
    n = len(vals)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = round(pos / (n - 1), 6)
    return out


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk, ranked within the batch."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        raws = [_raw_decision(m, extract_group_features(c)) for c in chunks]
        return _rank_normalize(raws)
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback (bounded sigmoid). The batch path is score_batch."""
    try:
        if not chunk:
            return 0.5
        z = _raw_decision(_model(), extract_group_features(chunk))
        return round(1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z)))), 6)
    except Exception:
        return 0.5
