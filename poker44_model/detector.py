"""Poker44 bot detector (family D) — tree ranker over the 180 C2 features.

score_batch(chunks) -> list[float] and score_chunk(chunk) -> float, each in
[0,1] (higher = more bot-like). Loads model.joblib and featurizes each chunk
via features.chunk_features. Inference does NOT sanitize: live chunks arrive
already sanitized by the validator (prepare_hand_for_miner runs validator-side).

Output = within-batch rank mapped through a logistic with an explicit positive-
fraction cap: convert each chunk's raw model score to its within-batch rank
u in [0,1], then score = sigmoid(TEMP*(u-(1-BOT_FRACTION))). This crosses 0.5 at
exactly the top BOT_FRACTION of each batch (bounded human FPR) and, being rank-
based, is level-invariant on the OOD live feed. A hard floor guarantees at least
one chunk always crosses 0.5 (never a hard zero).
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

BOT_FRACTION = 0.12   # top fraction of each batch mapped above 0.5 (positive cap)
TEMP = 22.0           # logistic steepness in rank space

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        m = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        try:
            m.set_params(n_jobs=1)
        except Exception:
            pass
        _MODEL = m
    return _MODEL


def _rank01(vals):
    n = len(vals)
    if n <= 1:
        return np.array([1.0] * n)
    order = np.argsort(np.argsort(np.asarray(vals, dtype=float), kind="mergesort"))
    return order / (n - 1)


def _decision(vals):
    u = _rank01(vals)
    scores = 1.0 / (1.0 + np.exp(-TEMP * (u - (1.0 - BOT_FRACTION))))
    # HARD-ZERO GUARD: guarantee >=1 chunk always crosses 0.5.
    if scores.size and float(np.max(scores)) < 0.5:
        scores[int(np.argmax(u))] = 0.5
    return [round(float(s), 6) for s in scores]


def _raw_scores(model, chunks):
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    X = np.asarray(rows, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.asarray(model.predict(X), dtype=float)


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk (rank + positive-fraction cap)."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        return _decision(list(_raw_scores(_model(), chunks)))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback; score_batch is the real entry point."""
    try:
        if not chunk:
            return 0.5
        return round(float(1.0 / (1.0 + np.exp(-float(_raw_scores(_model(), [chunk])[0])))), 6)
    except Exception:
        return 0.5
