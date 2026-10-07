"""Top-k extraction and coverage-aware re-ranking (shared by train_final.py and coverage_curve.py).

Coverage-aware re-ranking (popularity-bias control, cf. Abdollahpouri et al. RecSys 2017;
greedy re-ranking as in Steck, Calibrated Recommendations, RecSys 2018): the metric adds
0.2 * (#distinct predicted songs) / D.  Starting from the top-5 by probability, we swap a slot
whose song is also predicted in other sessions (so removing it here does not lower the distinct
count) for this session's best never-predicted alternative, cheapest expected-DCG loss first,
until the distinct-song count reaches the target.  Slots are processed from the lowest weight
(5th) upwards.
"""
import numpy as np
import polars as pl
from metric import W


def topk(cand, nq, k):
    """Per query: top-k candidates by p -> (songs[nq,k], probs[nq,k]) padded with -1 / 0."""
    s = (cand.sort(["qid", "p"], descending=[False, True]).group_by("qid", maintain_order=True)
             .head(k).group_by("qid", maintain_order=True).agg("cand", "p"))
    songs = np.full((nq, k), -1, np.int64)
    probs = np.zeros((nq, k), np.float64)
    for q, c, p in zip(s["qid"].to_list(), s["cand"].to_list(), s["p"].to_list()):
        songs[q, :len(c)] = c
        probs[q, :len(p)] = p
    return songs, probs


def fallback_for(queries):
    out = []
    for qs in queries:
        f = list(dict.fromkeys(int(x) for x in qs[::-1]))
        out.append(f + [-2 - i for i in range(5)])  # negative = dummy if the session is tiny
    return out


def coverage_rerank(songs, probs, target_u, fallback, slots=(4, 3)):
    nq = len(songs)
    pred, pp = songs[:, :5].copy(), probs[:, :5].copy()
    for r in np.nonzero((pred < 0).any(1))[0]:
        fill = [x for x in fallback[r] if x not in pred[r]]
        for c in range(5):
            if pred[r, c] < 0:
                pred[r, c], pp[r, c] = fill.pop(0), 0.0
    used = {}
    for x in pred.reshape(-1):
        used[x] = used.get(x, 0) + 1
    u = len(used)
    nxt = np.full(nq, 5)  # next alternative index to try per session
    for slot in slots:
        if u >= target_u:
            break
        props = []
        for r in range(nq):
            for a in range(nxt[r], songs.shape[1]):
                x = songs[r, a]
                if x < 0:
                    break
                if x not in used and x not in pred[r]:
                    props.append((W[slot] * (pp[r, slot] - probs[r, a]), r, a))
                    break
        props.sort()
        for loss, r, a in props:
            if u >= target_u:
                break
            old, x = pred[r, slot], songs[r, a]
            if used.get(old, 0) <= 1 or x in used:
                continue
            used[old] -= 1
            used[x] = 1
            pred[r, slot], pp[r, slot] = x, probs[r, a]
            nxt[r] = a + 1
            u += 1
    return pred, u
