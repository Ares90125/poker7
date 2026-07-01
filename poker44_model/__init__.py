"""Participant-owned model package for the Poker44 miner (uid7).

Bot detector = ExtraTrees + HistGradientBoosting ensemble over the v4 feature set
(v3 behavioral features — entropy + cross-hand duplication signatures + dispersion —
CONCAT sequence-aware features from seq_features.py: action n-grams, transition-matrix
entropy, actor-alternation, bet-sizing runs, street-progression), scored by
within-batch ranking. See detector.py (inference), features.py (extraction),
seq_features.py (sequence features), train_model.py (training), model.joblib.
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
