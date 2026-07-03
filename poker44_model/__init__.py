"""Participant-owned model package for the Poker44 miner (uid7) — v6_da CORAL.

model_name: poker7-da-v6

Bot detector = ExtraTrees(n_jobs=1) + HistGradientBoosting soft-vote ensemble
over the 180-feature C2 behavioral feature set, trained with CORAL domain
adaptation: benchmark-train features are aligned (mean + covariance) to the
UNLABELED live feature distribution before the classifier is fit, so the model
operates in the live feature space and its raw predict_proba does not collapse
on the live population (see detector.py / train_model.py). Inference does NOT
re-sanitize (live hands are already sanitized validator-side) and does NOT
re-apply the CORAL transform (live is already in the aligned space); scores are
within-batch ranks. See features.py (extraction + FEATURE_NAMES),
train_model.py (CORAL training), coral_transform.npz (baked mu_src/mu_tgt/W),
model.joblib (trained model).
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
