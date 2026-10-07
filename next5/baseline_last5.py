"""Baseline: the 5 most recent distinct songs of each test session (repeat-consumption prior).
Also evaluates the same rule on the validation queries."""
import numpy as np
import polars as pl
from common import DATA, WORK
from metric import report


def last5(seqs):
    out = np.full((len(seqs), 5), -1, np.int64)
    for r, s in enumerate(seqs):
        got = []
        for x in s[::-1]:
            if x not in got:
                got.append(x)
            if len(got) == 5:
                break
        out[r, :len(got)] = got
    return out


report(last5(np.load(WORK / "val_queries.npy")), np.load(WORK / "val_truth.npy"), tag="val last5-distinct")

songs = pl.read_parquet(WORK / "songs.parquet")["song_id"].to_numpy()
sid = np.load(WORK / "test_sid.npy")
pred = last5(np.load(WORK / "test_seq.npy"))
# sessions with <5 distinct songs: pad with globally popular songs not already predicted
pop = np.bincount(np.load(WORK / "train_seq.npy").reshape(-1)).argsort()[::-1][:50]
for r in np.nonzero((pred < 0).any(1))[0]:
    fill = [p for p in pop if p not in pred[r]]
    for c in range(5):
        if pred[r, c] < 0:
            pred[r, c] = fill.pop(0)
sub = pl.DataFrame({"session_id": sid, **{f"top{i+1}": songs[pred[:, i]] for i in range(5)}})
order = pl.read_csv(DATA / "sample.csv").select("session_id")
order.join(sub, on="session_id", how="left").write_csv(WORK / "sub_last5.csv")
print("written", sub.height)
