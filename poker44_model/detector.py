"""Poker44 bot detector -- UNIFIED BAG over the 180-dim sanitization-invariant
C2 feature row. Supports two sub types, averaged by WITHIN-BATCH rank:

  * "ens"  sub -- rank-fused 3-member ensemble (STACK LGBM+XGB+RF->logistic |
                  sign-stable monotone-LGBM bag | PCA->MLP bag), weights 0.40/0.25/0.35.
  * "vote" sub -- ET(700,d9)+RF(700,d9)+HGB(700,d9) soft-vote 0.45/0.25/0.30
                  (the u89/hot-poker3 published recipe, retrained on our rows).

Why a bag. Live seed-variance is ~0 in-benchmark (eval-rank Spearman 0.998) but
LARGE on the sanitized OOD live feed (fresh-model live fused-rank Spearman 0.85
same-recipe / 0.68 cross-recipe). The live seed noise is ~independent, so
averaging the within-batch ranks of M distinct-seed / cross-recipe subs cuts the
live per-round ranking jitter ~1/sqrt(M) with no measured floor. Mixing the two
recipes decorrelates harder (0.68 vs 0.85 live) -> more variance reduction per sub
AND inherits the vote's higher central discrimination.

Decision layer = the deployed STRICTLY-MONOTONE, tie-free NOISO layer (no isotonic;
anchor quantile Q + reward-fit logit margin/temp; FLOOR lifts exactly ceil(FLOOR*n)
chunks above 0.5, CAP pushes the rest below) -> deterministic crossing count,
hard-zero-safe by construction, monotone so the 65% rank block is set purely by the
bag's ordering. Inference does NOT sanitize (validator sanitizes live chunks).
All estimators pinned single-thread; HGB clamped via threadpool_limits at call time.
"""
from __future__ import annotations
import os
import numpy as np
import joblib
from poker44_model.features import chunk_features, FEATURE_NAMES

try:
    import torch  # noqa: F401
    torch.set_num_threads(1)
except Exception:
    pass
try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

_MODEL = None
_T_HI = 4e-4
_T_LO = -4e-4


def _pin_single_thread(est):
    for attr in ("n_jobs", "nthread", "thread_count"):
        try:
            est.set_params(**{attr: 1})
        except Exception:
            pass
    for holder in ("estimators_", "estimators"):
        try:
            for sub in getattr(est, holder):
                _pin_single_thread(sub[1] if isinstance(sub, tuple) else sub)
        except Exception:
            pass
    for attr in ("final_estimator_", "final_estimator"):
        try:
            _pin_single_thread(getattr(est, attr))
        except Exception:
            pass
    try:
        for _, step in est.steps:
            _pin_single_thread(step)
    except Exception:
        pass


def _model():
    global _MODEL
    if _MODEL is None:
        b = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))
        for sub in b["subs"]:
            for key in ("stack", "mono", "mlp", "et", "rf", "hgb"):
                if key in sub:
                    try:
                        _pin_single_thread(sub[key])
                    except Exception:
                        pass
        _MODEL = b
    return _MODEL


def _rank01(s):
    s = np.asarray(s, dtype=float)
    if s.size <= 1:
        return np.zeros_like(s)
    return np.argsort(np.argsort(s, kind="stable"), kind="stable").astype(float) / (s.size - 1)


def _rows(chunks):
    rows = []
    for c in chunks:
        feats = chunk_features(c)
        rows.append([feats.get(k, 0.0) for k in FEATURE_NAMES])
    return np.array(rows, dtype=float)


def _sub_rank(sub, X):
    """Within-batch rank produced by one sub (ens: member-rank blend; vote: vote-rank)."""
    if sub["kind"] == "ens":
        w1, w2, w3 = sub["weights"]
        s1 = sub["stack"].predict_proba(X)[:, 1]
        s2 = sub["mono"].predict_proba(X)[:, 1]
        s3 = sub["mlp"].predict_proba(X)[:, 1]
        return (w1 * _rank01(s1) + w2 * _rank01(s2) + w3 * _rank01(s3)) / (w1 + w2 + w3)
    w = sub["vote_weights"]
    p = (w[0] * sub["et"].predict_proba(X)[:, 1]
         + w[1] * sub["rf"].predict_proba(X)[:, 1]
         + w[2] * sub["hgb"].predict_proba(X)[:, 1]) / sum(w)
    return _rank01(p)


def _bag_fused(model, chunks):
    X = _rows(chunks)
    def _run():
        acc = None
        for sub in model["subs"]:
            r = _sub_rank(sub, X)
            acc = r if acc is None else acc + r
        return acc / len(model["subs"])
    if threadpool_limits is None:   # clamp HGB's OpenMP at call time
        return _run()
    with threadpool_limits(limits=1):
        return _run()


def _logit(p, eps):
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _decision(model, fused):
    eps = float(model["EPS"]); q = float(model["Q"]); margin = float(model["MARGIN"])
    temp = float(model.get("TEMP", 1.0)); floor = float(model["FLOOR"]); cap = bool(model.get("CAP", False))
    tref = float(model["train_ref_logit"]) - margin
    z = _logit(fused, eps)
    if z.size == 0:
        return []
    anchor = np.quantile(z, q)
    t = (z - anchor + tref) / temp
    order = np.argsort(-z, kind="mergesort")
    k = max(1, int(np.ceil(floor * len(t))))
    top, rest = order[:k], order[k:]
    d = _T_HI - t[top].min()
    if d > 0.0:
        t[top] = t[top] + d
    if cap and rest.size:
        d = t[rest].max() - _T_LO
        if d > 0.0:
            t[rest] = t[rest] - d
    return [round(float(s), 9) for s in 1.0 / (1.0 + np.exp(-t))]


def score_batch(chunks):
    """One bot-risk score in [0,1] per chunk (bag rank-fused, strictly-monotone NOISO)."""
    chunks = chunks or []
    if not chunks:
        return []
    try:
        m = _model()
        return _decision(m, _bag_fused(m, chunks))
    except Exception:
        return [0.5] * len(chunks)


def score_chunk(chunk):
    """Single-chunk fallback (no batch context): bag member-mean probability."""
    try:
        if not chunk:
            return 0.5
        m = _model()
        X = _rows([chunk])
        acc = 0.0
        for sub in m["subs"]:
            if sub["kind"] == "ens":
                w = sub["weights"]
                acc += (w[0] * sub["stack"].predict_proba(X)[:, 1]
                        + w[1] * sub["mono"].predict_proba(X)[:, 1]
                        + w[2] * sub["mlp"].predict_proba(X)[:, 1]) / sum(w)
            else:
                w = sub["vote_weights"]
                acc += (w[0] * sub["et"].predict_proba(X)[:, 1]
                        + w[1] * sub["rf"].predict_proba(X)[:, 1]
                        + w[2] * sub["hgb"].predict_proba(X)[:, 1]) / sum(w)
        return round(float(acc[0] / len(m["subs"])), 6)
    except Exception:
        return 0.5
