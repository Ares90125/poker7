"""Poker44 bot detector — uid7 (Ares90125/poker7).

Model: **ExtraTrees + HistGradientBoosting soft-vote ensemble** over the v3
behavioral feature set (entropy, cross-hand duplication signatures, run-length
regularity, quantile aggregates — see features.py). Output = **within-batch
rank**, which is robust to the benchmark-vs-live distribution shift and matches
the validator's ranking-based reward.

The trained model is the committed `model.joblib` (reproducible via
train_model.py against the public benchmark). sklearn loads it at inference.
`score_batch(chunks)` returns one rank-based bot-risk score in [0,1] per chunk.
"""
from __future__ import annotations

import os

import numpy as np
import joblib

from poker44_model.features import chunk_features_v4, FEATURE_NAMES

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
    return _MODEL


def _rank_normalize(vals):
    n = len(vals)
    if n <= 1:
        return [0.5] * n
    order = sorted(range(n), key=lambda i: vals[i])
    out = [0.0] * n
    for pos, i in enumerate(order):
        out[i] = round(pos / (n - 1), 6)
    return out


def _raw_scores(model, chunks):
    rows = []
    for c in chunks:
        feats = chunk_features_v4(c)       # v3 + seq feature sets, each computed ONCE per chunk
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return model.predict_proba(np.array(rows, dtype=float))[:, 1]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk, ranked within the batch."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        return _rank_normalize(list(_raw_scores(_model(), chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk model probability (fallback; batch path is score_batch)."""
    try:
        if not chunk:
            return 0.5
        return round(float(_raw_scores(_model(), [chunk])[0]), 6)
    except Exception:
        return 0.5
