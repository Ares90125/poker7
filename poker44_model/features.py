"""Behavioral feature extraction over a chunk GROUP (list of ~30 hands).

Uses ONLY fields present in the live miner-visible payload. Note the live
sanitizer zeroes `outcome` (no showdown/pot), strips hole/board cards, fixes
bb=0.02, and shows only a sampled action window — so the signal here is action
types, bet-sizing relative to pot, pot dynamics, stacks, and street depth.
"""
from __future__ import annotations

import math
from collections import Counter

ACTION_TYPES = ("fold", "check", "call", "bet", "raise")
STREETS = ("preflop", "flop", "turn", "river")
_CANON = (0.33, 0.5, 0.66, 0.75, 1.0, 1.25, 1.5, 2.0)


def _pstd(xs):
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def hand_features(h):
    acts = h.get("actions") or []
    players = h.get("players") or []
    cnt = Counter(a.get("action_type") for a in acts)
    n = max(1, sum(cnt.get(k, 0) for k in ACTION_TYPES))
    streets_seen = {str(a.get("street") or "").lower() for a in acts}
    depth = sum(1 for s in STREETS if s in streets_seen)
    aggr = cnt.get("bet", 0) + cnt.get("raise", 0)
    passive = cnt.get("call", 0) + cnt.get("check", 0)

    sizes = []
    betpot = []
    last_pot = 0.0
    pre_a = pre_n = post_a = post_n = limp = 0
    for a in acts:
        t = a.get("action_type")
        s = str(a.get("street") or "").lower()
        amt = float(a.get("normalized_amount_bb") or 0.0)
        pf = float(a.get("pot_before") or 0.0)
        pa = float(a.get("pot_after") or 0.0)
        last_pot = max(last_pot, pa)
        is_aggr = 1 if t in ("bet", "raise") else 0
        if s == "preflop":
            pre_n += 1; pre_a += is_aggr
            if t == "call":
                limp += 1
        else:
            post_n += 1; post_a += is_aggr
        if t in ("bet", "raise"):
            sizes.append(amt)
            if pf > 0:
                betpot.append((amt * 0.02) / pf)
    cluster = [min(abs(b - c) for c in _CANON) for b in betpot] if betpot else []
    distinct = (len(set(round(x, 1) for x in sizes)) / len(sizes)) if sizes else 0.0
    bm = (sum(sizes) / len(sizes)) if sizes else 0.0
    bs = _pstd(sizes)
    stacks = [float(p.get("starting_stack") or 0.0) for p in players]

    return {
        "n_act": float(len(acts)),
        "depth": float(depth),
        "fold_r": cnt.get("fold", 0) / n,
        "check_r": cnt.get("check", 0) / n,
        "call_r": cnt.get("call", 0) / n,
        "bet_r": cnt.get("bet", 0) / n,
        "raise_r": cnt.get("raise", 0) / n,
        "aggr_f": aggr / (passive + 1.0),
        "n_players": float(len(players)),
        "stack_mean": (sum(stacks) / len(stacks)) if stacks else 0.0,
        "stack_std": _pstd(stacks),
        "bet_mean": bm,
        "bet_std": bs,
        "bet_max": max(sizes) if sizes else 0.0,
        "betpot_mean": (sum(betpot) / len(betpot)) if betpot else 0.0,
        "betpot_std": _pstd(betpot),
        "final_pot_bb": last_pot / 0.02,
        "n_sizes": float(len(sizes)),
        "pre_aggr_r": pre_a / max(1, pre_n),
        "post_aggr_r": post_a / max(1, post_n),
        "limp_r": limp / max(1, pre_n),
        "betpot_cluster": (sum(cluster) / len(cluster)) if cluster else 0.5,
        "bet_cov": (bs / bm) if bm > 1e-6 else 0.0,
        "distinct_sizes": distinct,
    }


PERHAND = ["n_act", "depth", "fold_r", "check_r", "call_r", "bet_r", "raise_r",
           "aggr_f", "n_players", "stack_mean", "stack_std", "bet_mean", "bet_std",
           "bet_max", "betpot_mean", "betpot_std", "final_pot_bb", "n_sizes",
           "pre_aggr_r", "post_aggr_r", "limp_r", "betpot_cluster", "bet_cov",
           "distinct_sizes"]


def extract_group_features(group):
    """Return {feature_name: value}: mean & std of each per-hand feature."""
    rows = [hand_features(h) for h in group]
    feats = {}
    for k in PERHAND:
        vals = [r[k] for r in rows]
        feats[f"{k}_m"] = (sum(vals) / len(vals)) if vals else 0.0
        feats[f"{k}_s"] = _pstd(vals)
    feats["hands"] = float(len(group))
    return feats


FEATURE_NAMES = [f"{k}_m" for k in PERHAND] + [f"{k}_s" for k in PERHAND] + ["hands"]
