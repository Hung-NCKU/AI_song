"""Faithful port of the 1st-place "only_ngrams" solution (Vivaldi - My_cat_can_turn_some),
official Private 0.55569 / Public 0.5544, onto our validation split so both methods are compared
under identical conditions.  Source: github.com/afan0918/KKCompany-Music-Challenge-Next-5-Songcraft
(interesting/only_ngrams.ipynb).

Their n-gram tables: context of L songs (L = 4, 3, 2), each song at offset 1..5 after the context
adds weight [1, .63, .5, .43, .38][offset-1].  Prediction cascades through contexts
4321, 321, 21, 5432, 432, 6543, 543, 32 (song positions counted from the end), then pads with the
session's most recent songs and 5 fixed popular songs.  Ties keep first-insertion order like
collections.Counter.most_common.  Omitted: the "predict-from-predictions" stage, which their own
log shows fills only ~0.09% of slots.

Usage: python winner_ngram.py val|test
"""
import sys
import time
import numpy as np
import polars as pl
from common import WORK
from features import corpus_for
from metric import report

MODE = sys.argv[1]
W = [1, 0.63, 0.5, 0.43, 0.38]
t0 = time.time()
train_sid, test_sid = np.load(WORK / "train_sid.npy"), np.load(WORK / "test_sid.npy")
FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
corpus, queries = corpus_for(MODE, FRAC)
# global corpus order = sessions sorted by session_id (their all_data.sort_values('session_id'))
train = np.load(WORK / "train_seq.npy")
rng = np.random.default_rng(0)
perm = rng.permutation(len(train))
n_val = len(train) // 5
val_idx, fit_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
if FRAC < 1.0:
    fit_idx = fit_idx[np.random.default_rng(2).random(len(fit_idx)) < FRAC]
sids = [train_sid[fit_idx], train_sid[val_idx], test_sid] if MODE == "val" else [train_sid, test_sid]
rank_of = {s: r for r, s in enumerate(np.sort(np.concatenate(sids)))}
nq = len(queries)

# (stage name, table L, context = queries[:, a:b])
STAGES = [("4321", 4, 16, 20), ("321", 3, 17, 20), ("21", 2, 18, 20), ("5432", 4, 15, 19),
          ("432", 3, 16, 19), ("6543", 4, 14, 18), ("543", 3, 15, 18), ("32", 2, 17, 19)]
cc = lambda L: [f"c{i}" for i in range(L)]
lists = {}
for L in (4, 3, 2):
    need = pl.concat([pl.DataFrame({f"c{m}": queries[:, a + m] for m in range(L)})
                      for (_, l, a, b) in STAGES if l == L]).unique()
    parts = []
    for A, S in zip(corpus, sids):
        srank = np.array([rank_of[s] for s in S], np.int64)
        T = A.shape[1]
        for i in range(0, T - L):
            for off in range(1, 6):
                if i + L - 1 + off >= T:
                    break
                df = pl.DataFrame({**{f"c{m}": A[:, i + m] for m in range(L)},
                                   "nxt": A[:, i + L - 1 + off],
                                   "w": np.full(len(A), W[off - 1], np.float64),
                                   "ord": srank * 64 + i * 8 + off})
                parts.append(df.join(need, on=cc(L), how="semi"))
    p = pl.concat(parts)
    agg = (p.group_by(cc(L) + ["nxt"]).agg(pl.col("w").sum(), pl.col("ord").min())
             .sort(["w", "ord"], descending=[True, False])
             .group_by(cc(L), maintain_order=True).agg(pl.col("nxt").head(15)))
    for (name, l, a, b) in STAGES:
        if l != L:
            continue
        q = pl.DataFrame({"qid": np.arange(nq), **{f"c{m}": queries[:, a + m] for m in range(L)}})
        lists[name] = dict(zip(*q.join(agg, on=cc(L)).select("qid", "nxt").to_dict(as_series=False).values()))
    print(f"L={L} done t={time.time() - t0:.0f}s", flush=True)

FILL = None
if True:  # their 5 hard-coded padding songs, mapped to our integer ids (if present)
    songs = pl.read_parquet(WORK / "songs.parquet").select("song_idx", "song_id")
    m = dict(zip(songs["song_id"].to_list(), songs["song_idx"].to_list()))
    FILL = [m[s] for s in ["18a62aea3e0e67e21ea56c125c29c474", "85422f927d88358292985cb319d216fa",
                           "ef71212934e35b400232a0cd8e7e67a2", "4a27803802d6090389507671d8aed6eb",
                           "bc58d186329eda8b23e510ba98b2c1ea"] if s in m]

pred = np.full((nq, 5), -1, np.int64)
for r in range(nq):
    qs = queries[r]
    out = []
    if qs[-1] == qs[-2]:
        out.append(int(qs[-1]))
    for name, *_ in STAGES:
        if len(out) >= 5:
            break
        for s in lists[name].get(r, []):
            if s not in out:
                out.append(s)
            if len(out) >= 5:
                break
    for i in range(1, 21):
        if len(out) >= 5:
            break
        if qs[-i] not in out:
            out.append(int(qs[-i]))
    for s in FILL:
        if len(out) >= 5:
            break
        out.append(s)
    pred[r] = out[:5]
if FRAC == 1.0:
    np.save(WORK / f"winner_pred_{MODE}.npy", pred)
print(f"done t={time.time() - t0:.0f}s")
if MODE == "val":
    truth = np.load(WORK / "val_truth.npy")
    half = np.random.default_rng(1).random(nq) < 0.5
    report(pred, truth, tag="1st-place only_ngrams (all val)")
    report(pred[~half], truth[~half], tag="1st-place only_ngrams (val half B)")
