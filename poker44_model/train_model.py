"""Reproducible training for uid7 → writes model.joblib.

v3 features (entropy + cross-hand signatures + dispersion) + an
ExtraTrees + HistGradientBoosting soft-vote ensemble, trained on ALL labeled
public-benchmark examples. Requires sklearn (`pip install -e .`).

    python3 poker44_model/train_model.py --data /root/ares/Poker/train/raw
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import joblib
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              VotingClassifier)

from poker44_model.features import chunk_features, FEATURE_NAMES


def load(raw):
    out = []
    for f in sorted(glob.glob(os.path.join(raw, "chunks_*.json"))):
        for rc in json.load(open(f)).get("chunks", []):
            for g, l in zip(rc.get("chunks") or [], rc.get("groundTruth") or []):
                out.append((g, int(l)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to train/raw chunk JSON dir")
    args = ap.parse_args()

    data = load(args.data)
    rows, y = [], []
    for g, l in data:
        feats = chunk_features(g)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
        y.append(l)
    X = np.array(rows, dtype=float)
    y = np.array(y)

    model = VotingClassifier([
        ('et', ExtraTreesClassifier(n_estimators=300, min_samples_leaf=4, random_state=0, n_jobs=-1)),
        ('h', HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03, max_iter=300,
                                             l2_regularization=1.0, random_state=0)),
    ], voting='soft').fit(X, y)

    out = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, out)
    print(f"wrote {out} ({len(data)} examples, {len(FEATURE_NAMES)} features)")


if __name__ == "__main__":
    main()
