"""Participant-owned model package for the Poker44 miner (uid7).

Bot detector = logistic regression over winsorized behavioral features, scored
by within-batch ranking (robust to the benchmark-vs-live distribution shift).
See detector.py (inference), features.py (extraction), train_model.py (training),
model.json (parameters).
"""

from poker44_model.detector import score_batch, score_chunk

__all__ = ["score_batch", "score_chunk"]
