"""Participant-owned model package for the Poker44 miner — pure-tree `poker-xgb-tuned`.

Bot detector = a single tuned XGBoost GBDT (hist, 800 trees, depth 6, lr 0.02,
subsample/colsample 0.8, L1+L2) over C2's 180 sanitization-invariant features.
PURE tree head (no linear head — the new-eval live signal favors pure trees).
Trained on benchmark hands passed through the validator's prepare_hand_for_miner
so training matches the sanitized live feed (train==serve); scored by
within-batch ranking. Inference does NOT re-sanitize (live hands are already
sanitized validator-side). See detector.py (inference), features.py (extraction
+ FEATURE_NAMES), train_model.py (training), model.joblib (trained model).
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
