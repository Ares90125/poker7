"""Poker44 bot detector — pure-tree candidate `poker-xgb-tuned`.

Model: a single **tuned XGBoost gradient-boosted tree** (hist tree method,
n_estimators=800, max_depth=6, learning_rate=0.02, subsample/colsample 0.8,
L1+L2 regularized) over C2's 180 sanitization-invariant features. This is a
PURE GBDT — there is NO linear head. The new-eval live signal (2026-07-07)
showed the L1-logistic head in the linblend miners consistently costs ~0.15
reward vs pure trees, so this candidate deliberately keeps a tree-only head,
distinct in inductive bias from the LightGBM and from C2's ExtraTrees+HGB vote.

IMPORTANT — inference does NOT sanitize. Live chunks arrive already sanitized by
the validator (prepare_hand_for_miner runs validator-side, per hand). Only
TRAINING sanitizes raw benchmark hands (see train_model.py). Featurizing the
incoming chunks directly keeps train==serve.

Output = **within-batch rank**, matching the validator's ranking-based reward.
`score_batch(chunks)` returns one score in [0,1] per chunk. n_jobs is pinned to
1 everywhere (n_jobs=-1 deadlocks the axon on batched predict).
"""
from __future__ import annotations

import os

import numpy as np
import joblib

from poker44_model.features import chunk_features, FEATURE_NAMES

_MODEL = None


def _model():
    global _MODEL
    if _MODEL is None:
        _MODEL = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        # belt-and-suspenders: never let a deserialized booster fan out threads
        try:
            _MODEL.set_params(n_jobs=1)
        except Exception:
            pass
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
    # Live chunks are already sanitized by the validator; featurize as-is.
    rows = []
    for c in chunks:
        feats = chunk_features(c)          # compute the feature set ONCE per chunk
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
