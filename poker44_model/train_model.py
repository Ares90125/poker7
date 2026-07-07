"""Reproducible training for the pure-tree `poker-xgb-tuned` -> writes model.joblib.

A single tuned XGBoost GBDT over C2's 180 sanitization-invariant features
(features.py FEATURE_NAMES). PURE tree head, no linear head.

Every raw benchmark hand is passed through the validator's
`prepare_hand_for_miner` (the anti-leakage / canonicalization sanitizer, from
poker44/validator/payload_view.py) BEFORE feature extraction, so the training
distribution matches what the validator serves miners (train==serve). Live
chunks are already sanitized validator-side, so inference does NOT re-sanitize.

    python3 poker44_model/train_model.py --data /root/ares/Poker/train/raw \
        --payload-view /root/ares/Poker/main/poker44/validator/payload_view.py
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import typing

import numpy as np
import joblib
from xgboost import XGBClassifier

from poker44_model.features import chunk_features, FEATURE_NAMES

# Tuned pure-GBDT hyperparameters (n_jobs=1 — n_jobs=-1 deadlocks the axon).
XGB_PARAMS = dict(
    n_estimators=800,
    max_depth=6,
    learning_rate=0.02,
    tree_method="hist",
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=4,
    gamma=0.0,
    reg_lambda=2.0,
    reg_alpha=0.5,
    n_jobs=1,
    random_state=0,
    eval_metric="logloss",
)


def make_model():
    return XGBClassifier(**XGB_PARAMS)


def _load_sanitizer(pv_path):
    spec = importlib.util.spec_from_file_location("_p44_payload_view", pv_path)
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)
    pv.Optional = typing.Optional  # payload_view uses Optional but never imports it
    fn = pv.prepare_hand_for_miner

    def sanitize_chunk(chunk):
        out = []
        for h in (chunk or []):
            try:
                out.append(fn(h))
            except Exception:
                out.append(h)
        return out

    return sanitize_chunk


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
    ap.add_argument("--payload-view", required=True,
                    help="path to poker44/validator/payload_view.py (the sanitizer)")
    args = ap.parse_args()

    sanitize_chunk = _load_sanitizer(args.payload_view)

    data = load(args.data)
    rows, y = [], []
    for g, l in data:
        feats = chunk_features(sanitize_chunk(g))   # TRAIN == SERVE: sanitize raw hands
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
        y.append(l)
    X = np.nan_to_num(np.array(rows, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y)

    model = make_model().fit(X, y)

    out = os.path.join(os.path.dirname(__file__), "model.joblib")
    joblib.dump(model, out)
    print(f"wrote {out} ({len(data)} examples, {len(FEATURE_NAMES)} features)")


if __name__ == "__main__":
    main()
