"""Poker44 bot detector — uid7 (Ares90125/poker7), v6_da candidate CORAL.

Model: ExtraTrees(n_jobs=1) + HistGradientBoosting soft-vote ensemble over the
same 180-feature C2 behavioral feature set. The upgrade over C2 (v5_sani) is
DOMAIN ADAPTATION baked into TRAINING: the ensemble was fit on benchmark
features CORAL-aligned to the UNLABELED live feature covariance (see
train_model.py / fit_coral). This re-centers and re-colors the 2nd-order
statistics of the benchmark-train features to the live population, so raw
predict_proba no longer collapses toward a constant on live chunks
(live raw-std 0.076 -> 0.133) and duplication-heavy chunks rank high
(dup-corr 0.078 -> 0.606). No labels are used for the alignment; benchmark
labels are used only for the classifier fit. No live labels exist.

IMPORTANT — inference does NOT sanitize and does NOT re-apply the CORAL
transform. Live chunks arrive already sanitized by the validator
(prepare_hand_for_miner runs validator-side) AND are already in the live
feature space the model was aligned to during training, so this path featurizes
the incoming chunks directly and calls the model as-is. Applying the transform
again at inference would double-shift already-live-space data. The baked
alignment (mu_src, mu_tgt, W) is shipped in coral_transform.npz for reference /
benchmark-space use only.

Output = within-batch rank in [0,1] (higher = more bot-like), matching the
validator's ranking-based reward. ExtraTrees n_jobs=1 (deterministic, single
thread). No thresholds / clips / rank tricks beyond the within-batch rank.
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
    # Live chunks are already sanitized AND already in the CORAL-aligned live
    # feature space; featurize as-is (no re-sanitize, no re-transform).
    rows = []
    for c in chunks:
        feats = chunk_features(c)
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
