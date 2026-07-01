"""Sequence-aware features for Candidate A.

Per-hand: action bigram/trigram frequencies, action-transition-matrix entropy,
actor-alternation patterns, bet-sizing run/monotonicity patterns, street-progression
n-grams. Then aggregate across the group (mean/std) and concat with v3 features.

Design keeps the feature count bounded (small data -> overfit risk) by using a fixed
vocabulary of action types and streets and a curated set of scalar per-hand summaries
rather than an explosive one-hot of every possible n-gram.
"""
from __future__ import annotations

import math
from collections import Counter

BB = 0.02

# Fixed vocabularies (verified against data)
ACTS = ("fold", "raise", "call", "check", "bet")
STREETS = ("preflop", "flop", "turn", "river")
ACT_IDX = {a: i for i, a in enumerate(ACTS)}


def _f(v, d=0.0):
    try:
        return d if v is None else float(v)
    except (TypeError, ValueError):
        return d


def _i(v, d=0):
    try:
        return int(v)
    except (TypeError, ValueError):
        return d


def _div(a, b):
    return a / b if b else 0.0


def _mean(xs):
    return _div(sum(xs), len(xs))


def _std(xs):
    if not xs:
        return 0.0
    m = _mean(xs)
    return math.sqrt(max(0.0, _mean([(x - m) ** 2 for x in xs])))


def _quant(xs, q):
    if not xs:
        return 0.0
    s = sorted(float(x) for x in xs)
    if len(s) == 1:
        return s[0]
    pos = min(max(q, 0.0), 1.0) * (len(s) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return s[lo] if lo == hi else s[lo] * (1 - (pos - lo)) + s[hi] * (pos - lo)


def _norm_entropy(counts):
    """Shannon entropy of a Counter/list of counts, normalized to [0,1]."""
    vals = list(counts)
    tot = float(sum(vals))
    if tot <= 0 or len(vals) <= 1:
        return 0.0
    nz = [v for v in vals if v > 0]
    if len(nz) <= 1:
        return 0.0
    e = -sum((v / tot) * math.log(v / tot) for v in nz)
    return e / math.log(len(vals))


# Curated bigrams that carry poker-line meaning (bounded set -> less overfit).
KEY_BIGRAMS = (
    ("check", "check"), ("check", "bet"), ("bet", "call"), ("bet", "raise"),
    ("bet", "fold"), ("raise", "fold"), ("raise", "call"), ("raise", "raise"),
    ("call", "check"), ("call", "call"), ("check", "raise"), ("call", "fold"),
)
KEY_TRIGRAMS = (
    ("check", "check", "check"), ("check", "bet", "call"), ("bet", "raise", "fold"),
    ("bet", "raise", "call"), ("raise", "raise", "fold"), ("check", "bet", "fold"),
)


def hand_seq_features(hand):
    actions = hand.get("actions") or []
    atypes, actors, snames, amts = [], [], [], []
    for a in actions:
        if not isinstance(a, dict):
            continue
        t = str(a.get("action_type") or "").lower().strip()
        atypes.append(t if t in ACT_IDX else "fold")
        actors.append(_i(a.get("actor_seat"), 0))
        s = str(a.get("street") or "").lower().strip()
        snames.append(s)
        amts.append(max(0.0, _f(a.get("normalized_amount_bb"))))

    n = len(atypes)
    nact = max(1.0, float(n))

    # --- bigram / trigram frequencies (curated key set) ---
    bigrams = list(zip(atypes, atypes[1:]))
    trigrams = list(zip(atypes, atypes[1:], atypes[2:]))
    bc = Counter(bigrams)
    tc = Counter(trigrams)
    nbig = max(1.0, float(len(bigrams)))
    ntri = max(1.0, float(len(trigrams)))

    out = {}
    for bg in KEY_BIGRAMS:
        out[f"big_{bg[0]}_{bg[1]}"] = _div(bc.get(bg, 0), nbig)
    for tg in KEY_TRIGRAMS:
        out[f"tri_{tg[0]}_{tg[1]}_{tg[2]}"] = _div(tc.get(tg, 0), ntri)

    # bigram/trigram diversity
    out["bigram_entropy"] = _norm_entropy(bc.values())
    out["bigram_unique_sh"] = _div(len(bc), nbig)
    out["trigram_unique_sh"] = _div(len(tc), ntri)

    # --- action-transition-matrix entropy ---
    # 5x5 transition matrix over action types; measure structure of the line.
    trans = [[0] * 5 for _ in range(5)]
    for a, b in bigrams:
        trans[ACT_IDX[a]][ACT_IDX[b]] += 1
    # global normalized entropy of the flattened transition matrix
    out["trans_matrix_entropy"] = _norm_entropy([c for row in trans for c in row])
    # mean conditional entropy of next-action given current action
    cond_ents = []
    for i in range(5):
        row = trans[i]
        if sum(row) > 0:
            cond_ents.append(_norm_entropy(row))
    out["trans_cond_entropy_mean"] = _mean(cond_ents)
    out["trans_cond_entropy_max"] = max(cond_ents) if cond_ents else 0.0
    # fraction of transitions that are self-loops (repeat same action)
    out["trans_selfloop_sh"] = _div(sum(trans[i][i] for i in range(5)), nbig)

    # --- actor-alternation patterns ---
    real_actors = [s for s in actors if s > 0]
    if len(real_actors) >= 2:
        switches = sum(1 for x, y in zip(real_actors, real_actors[1:]) if x != y)
        out["actor_alt_rate"] = _div(switches, len(real_actors) - 1)
        # longest run of same actor / heads-up style ping-pong detection
        longest = cur = 1
        for x, y in zip(real_actors, real_actors[1:]):
            cur = 1 if x != y else cur + 1
            longest = max(longest, cur)
        out["actor_run_max_sh"] = _div(longest, len(real_actors))
        # perfect-alternation (A,B,A,B...) score for heads-up
        if len(set(real_actors)) == 2:
            perfect = sum(1 for x, y in zip(real_actors, real_actors[1:]) if x != y)
            out["actor_pingpong"] = _div(perfect, len(real_actors) - 1)
        else:
            out["actor_pingpong"] = 0.0
    else:
        out["actor_alt_rate"] = 0.0
        out["actor_run_max_sh"] = 0.0
        out["actor_pingpong"] = 0.0

    # --- bet-sizing run / monotonicity patterns ---
    nz = [v for v in amts if v > 0]
    if len(nz) >= 2:
        incr = sum(1 for x, y in zip(nz, nz[1:]) if y > x + 1e-9)
        decr = sum(1 for x, y in zip(nz, nz[1:]) if y < x - 1e-9)
        eq = sum(1 for x, y in zip(nz, nz[1:]) if abs(y - x) <= 1e-9)
        d = max(1, len(nz) - 1)
        out["bet_incr_sh"] = _div(incr, d)
        out["bet_decr_sh"] = _div(decr, d)
        out["bet_eq_sh"] = _div(eq, d)
        out["bet_monotonic_up"] = 1.0 if incr == d else 0.0
        # longest monotonic-increasing run of bet sizes
        longest = cur = 1
        for x, y in zip(nz, nz[1:]):
            cur = cur + 1 if y > x + 1e-9 else 1
            longest = max(longest, cur)
        out["bet_up_run_max_sh"] = _div(longest, len(nz))
        # coefficient of variation of bet sizes (mechanical betting -> low CV)
        m = _mean(nz)
        out["bet_cv"] = _div(_std(nz), m) if m > 0 else 0.0
        # ratio-repeat: fraction of consecutive equal ratios (fixed-size betting)
        ratios = [round(_div(y, x), 3) for x, y in zip(nz, nz[1:]) if x > 0]
        if ratios:
            out["bet_ratio_top_sh"] = _div(max(Counter(ratios).values()), len(ratios))
        else:
            out["bet_ratio_top_sh"] = 0.0
    else:
        for k in ("bet_incr_sh", "bet_decr_sh", "bet_eq_sh", "bet_monotonic_up",
                  "bet_up_run_max_sh", "bet_cv", "bet_ratio_top_sh"):
            out[k] = 0.0

    # --- street-progression n-grams ---
    # compressed street sequence (collapse consecutive duplicates)
    street_seq = []
    for s in snames:
        if not street_seq or street_seq[-1] != s:
            street_seq.append(s)
    depth = sum(1 for s in ("flop", "turn", "river") if s in set(snames))
    out["street_depth"] = float(depth)
    out["reached_river"] = 1.0 if "river" in set(snames) else 0.0
    out["street_transitions"] = float(max(0, len(street_seq) - 1))
    # per-street action-count profile (how the line spreads across streets)
    sc = Counter(snames)
    for st in STREETS:
        out[f"street_{st}_sh"] = _div(sc.get(st, 0), nact)
    # entropy of action-count-across-streets distribution
    out["street_action_entropy"] = _norm_entropy([sc.get(st, 0) for st in STREETS])

    return out


# Determine the per-hand feature name list from an empty-ish hand
_PROBE = hand_seq_features({"actions": [
    {"action_type": "check", "actor_seat": 1, "street": "flop", "normalized_amount_bb": 1.0},
    {"action_type": "bet", "actor_seat": 2, "street": "flop", "normalized_amount_bb": 2.0},
]})
PERHAND_SEQ = sorted(_PROBE.keys())
_AGG = ("mean", "std", "q10", "q50", "q90")


def group_seq_features(group):
    """Aggregate per-hand sequence features across a group."""
    out = {}
    if not group:
        for name in PERHAND_SEQ:
            for agg in _AGG:
                out[f"seq_{name}_{agg}"] = 0.0
        return out
    rows = [hand_seq_features(h) for h in group]
    for name in PERHAND_SEQ:
        xs = [r.get(name, 0.0) for r in rows]
        out[f"seq_{name}_mean"] = _mean(xs)
        out[f"seq_{name}_std"] = _std(xs)
        out[f"seq_{name}_q10"] = _quant(xs, 0.1)
        out[f"seq_{name}_q50"] = _quant(xs, 0.5)
        out[f"seq_{name}_q90"] = _quant(xs, 0.9)
    return out


SEQ_FEATURE_NAMES = sorted(group_seq_features([
    {"actions": [{"action_type": "check", "actor_seat": 1, "street": "flop", "normalized_amount_bb": 1.0}]}
]).keys())
