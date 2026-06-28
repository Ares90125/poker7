"""Reproducible training for the uid7 logistic detector → writes model.json.

Requires the public benchmark (see ../train or the project's train/ folder) and
sklearn (`pip install -e .`). Usage from the repo root:

    python3 poker44_model/train_model.py --data /root/ares/Poker/train/raw

Trains on the benchmark `train` split. Evaluate on the `validation` split with
the subnet reward metric (0.75*AP + 0.25*recall@FPR<=0.05).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from poker44_model.features import extract_group_features, FEATURE_NAMES


def load(raw, split):
    out = []
    for f in sorted(glob.glob(os.path.join(raw, "chunks_*.json"))):
        for rc in json.load(open(f)).get("chunks", []):
            if rc.get("split") != split:
                continue
            for g, l in zip(rc.get("chunks") or [], rc.get("groundTruth") or []):
                out.append((g, int(l)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to train/raw chunk JSON dir")
    args = ap.parse_args()

    tr = load(args.data, "train")
    X = np.array([[extract_group_features(g)[k] for k in FEATURE_NAMES] for g, _ in tr])
    y = np.array([l for _, l in tr])

    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=0.5, max_iter=3000).fit(scaler.transform(X), y)

    model = {
        "type": "logistic",
        "features": FEATURE_NAMES,
        "mean": [round(float(v), 6) for v in scaler.mean_],
        "scale": [round(float(v), 6) for v in scaler.scale_],
        "coef": [round(float(v), 6) for v in clf.coef_[0]],
        "intercept": round(float(clf.intercept_[0]), 6),
    }
    out = os.path.join(os.path.dirname(__file__), "model.json")
    json.dump(model, open(out, "w"))
    print(f"wrote {out} ({len(tr)} train examples, {len(FEATURE_NAMES)} features)")


if __name__ == "__main__":
    main()
