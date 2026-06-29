"""Participant-owned model package for the Poker44 miner (uid7).

Bot detector = ExtraTrees + HistGradientBoosting ensemble over the v3 behavioral
feature set (entropy + cross-hand duplication signatures + dispersion), scored by
within-batch ranking. See detector.py (inference), features.py (extraction),
train_model.py (training), model.joblib (trained model).
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
