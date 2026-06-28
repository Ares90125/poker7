"""Participant-owned model package for the Poker44 miner (uid7).

Bot detector = logistic regression over behavioral features. See detector.py
(inference), features.py (feature extraction), train_model.py (reproducible
training), and model.json (learned parameters).
"""

from poker44_model.detector import score_chunk

__all__ = ["score_chunk"]
