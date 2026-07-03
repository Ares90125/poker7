"""v6_da candidate CORAL: covariance alignment (domain adaptation).

Idea. C2 (v5_sani) trains on validator-sanitized BENCHMARK hands and is scored on
a DIFFERENT live population. Benchmark features land OOD on live, so raw
predict_proba collapses toward a constant (near-random ranking). CORAL aligns the
2nd-order statistics of the two feature distributions: whiten the benchmark-train
feature covariance and re-color it to the LIVE feature covariance, THEN retrain
C2's exact ensemble on the aligned features. At inference the live chunks are
already in live feature space, so NO transform is applied (identity) — the model
was trained to operate directly in live coordinates.

    X_src_aligned = (X_src - mu_src) @ W  +  mu_tgt
    W = C_src^{-1/2} @ C_tgt^{1/2}      (both covariances Tikhonov-regularized)

We estimate C_tgt / mu_tgt from the UNLABELED captured live chunks (no labels
used — only feature covariance). Ensemble = ExtraTrees(n_jobs=1) + HistGBM
soft-vote, identical hyperparameters to C2. No thresholds / clips / rank tricks.

Bakes model.joblib + coral_transform.npz (mu_src, mu_tgt, W) so inference could
apply the transform if ever fed benchmark-space data; the live path uses identity.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
import sys
import typing

import numpy as np
import joblib
from scipy import linalg
from sklearn.ensemble import (ExtraTreesClassifier,
                              HistGradientBoostingClassifier,
                              VotingClassifier)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from features import chunk_features, FEATURE_NAMES  # noqa: E402

RAW_DIR = "/root/ares/Poker/train/raw"
PV_PATH = "/root/ares/Poker/main/poker44/validator/payload_view.py"
CAP_DIRS = [
    "/root/ares/Poker/Poker44-uid7/poker44_model/captures",
    "/root/ares/Poker/Poker44-uid73/poker44_model/captures",
]
EPS = 1e-6  # Tikhonov ridge on covariances (numerical stability)


# --- sanitizer (train==serve) ------------------------------------------------
def _load_sanitizer(pv_path):
    spec = importlib.util.spec_from_file_location("_p44_payload_view", pv_path)
    pv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pv)
    pv.Optional = typing.Optional
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


# --- data --------------------------------------------------------------------
def load_benchmark(raw):
    out = []
    for f in sorted(glob.glob(os.path.join(raw, "chunks_*.json"))):
        for rc in json.load(open(f)).get("chunks", []):
            for g, l in zip(rc.get("chunks") or [], rc.get("groundTruth") or []):
                out.append((g, int(l)))
    return out


def load_live_unique():
    """Unique live chunk sets deduped by top-level chunk_hash."""
    seen = {}
    for cd in CAP_DIRS:
        for f in glob.glob(os.path.join(cd, "*", "*", "chunks.json")):
            d = json.load(open(f))
            seen.setdefault(d["chunk_hash"], d["chunks"])
    live = []
    for h in sorted(seen):
        live.extend(seen[h])
    return live


def featurize(chunks):
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return np.asarray(rows, dtype=float)


# --- CORAL -------------------------------------------------------------------
def _sqrtm_psd(C):
    """Symmetric PSD matrix square root via eigendecomposition (stable)."""
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.T


def _inv_sqrtm_psd(C):
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, EPS, None)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def fit_coral(X_src, X_tgt):
    """Return (mu_src, mu_tgt, W) mapping source features into target 2nd-order
    statistics: X_aligned = (X_src - mu_src) @ W + mu_tgt."""
    d = X_src.shape[1]
    mu_src = X_src.mean(axis=0)
    mu_tgt = X_tgt.mean(axis=0)
    Cs = np.cov(X_src, rowvar=False) + EPS * np.eye(d)
    Ct = np.cov(X_tgt, rowvar=False) + EPS * np.eye(d)
    W = _inv_sqrtm_psd(Cs) @ _sqrtm_psd(Ct)
    return mu_src, mu_tgt, W


def apply_coral(X, mu_src, mu_tgt, W):
    return (X - mu_src) @ W + mu_tgt


# --- ensemble (identical to C2) ---------------------------------------------
def build_ensemble(seed=0):
    et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=4,
                              random_state=seed, n_jobs=1)  # baked n_jobs=1
    hgb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.03,
                                         max_iter=300, l2_regularization=1.0,
                                         random_state=seed)
    return VotingClassifier(estimators=[("et", et), ("hgb", hgb)], voting="soft")


def main():
    sanitize_chunk = _load_sanitizer(PV_PATH)

    # source: sanitized benchmark (labeled)
    data = load_benchmark(RAW_DIR)
    src_chunks = [sanitize_chunk(g) for g, _ in data]
    y = np.array([l for _, l in data])
    X_src = featurize(src_chunks)

    # target: unlabeled live captures (already sanitized) — covariance only
    live_chunks = load_live_unique()
    X_tgt = featurize(live_chunks)

    print(f"src {X_src.shape}  tgt {X_tgt.shape}  pos={int(y.sum())}")

    mu_src, mu_tgt, W = fit_coral(X_src, X_tgt)
    X_src_aligned = apply_coral(X_src, mu_src, mu_tgt, W)

    model = build_ensemble(seed=0).fit(X_src_aligned, y)

    joblib.dump(model, os.path.join(HERE, "model.joblib"))
    np.savez(os.path.join(HERE, "coral_transform.npz"),
             mu_src=mu_src, mu_tgt=mu_tgt, W=W,
             feature_names=np.array(FEATURE_NAMES))
    print("wrote model.joblib + coral_transform.npz")


if __name__ == "__main__":
    main()
