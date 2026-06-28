"""Poker44 bot detector — uid7 (Ares90125/poker7).

Model: **logistic regression** over standardized behavioral features
(see features.py). A transparent linear baseline. Learned parameters live in
model.json and are reproducible with train_model.py against the public
benchmark. Pure-Python at inference — no sklearn needed to serve.

`score_chunk(group)` returns P(bot) in [0, 1] for one chunk group.
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


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def score_chunk(chunk):
    """One bot-risk score in [0, 1] for a chunk group (list of hand dicts)."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        feats = extract_group_features(chunk)
        z = m["intercept"]
        for i, name in enumerate(m["features"]):
            scale = m["scale"][i] or 1.0
            z += m["coef"][i] * ((feats.get(name, 0.0) - m["mean"][i]) / scale)
        return round(_sigmoid(z), 6)
    except Exception:
        # Never crash the miner: a thrown exception = 0 coverage for the cycle.
        return 0.5
