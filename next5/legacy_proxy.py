"""Evaluate the previous approach of this repo (../model.py + ../score.py) on the same validation set.

The old pipeline set target = song frequency share (score.py), fit XGBRegressor on per-play
features, then for each test session ranked the session's OWN rows by predicted score and kept
the top 5.  Because the regression target is popularity, the result is essentially "the 5 most
popular songs the user already played in this session".  We evaluate that rule directly
(distinct songs, ranked by global play count), plus the simple "last 5 distinct" rule.
"""
import numpy as np
from common import WORK
from metric import report
from features import corpus_for

corpus, queries = corpus_for("val")
truth = np.load(WORK / "val_truth.npy")
pop = np.zeros(int(max(a.max() for a in corpus)) + 1)
for A in corpus:
    np.add.at(pop, A.reshape(-1), 1)

pred = np.zeros((len(queries), 5), np.int64)
for r, qs in enumerate(queries):
    u = list(dict.fromkeys(qs.tolist()))
    u.sort(key=lambda s: -pop[s])
    u += [-1 - i for i in range(5)]
    pred[r] = u[:5]
report(pred, truth, tag="legacy proxy: 5 most popular songs already in session")

last5 = np.zeros_like(pred)
for r, qs in enumerate(queries):
    u = list(dict.fromkeys(qs[::-1].tolist())) + [-1 - i for i in range(5)]
    last5[r] = u[:5]
report(last5, truth, tag="baseline: last 5 distinct songs")
