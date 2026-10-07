"""DCG / coverage trade-off of the coverage-aware re-ranking on validation (out-of-fold probabilities).
For each candidate denominator D, report the distinct-count target that maximises 0.8*DCG + 0.2*U/D."""
import numpy as np
import polars as pl
from common import WORK
from metric import dcg
from postproc import topk, coverage_rerank, fallback_for

oof = pl.read_parquet(WORK / "val_oof.parquet")
truth = np.load(WORK / "val_truth.npy")
queries = np.load(WORK / "val_queries.npy")
nq = len(truth)
songs, probs = topk(oof, nq, 25)
fb = fallback_for(queries)
u_win = len(np.unique(np.load(WORK / "winner_pred_val.npy")))
d_win = dcg(np.load(WORK / "winner_pred_val.npy"), truth)
rows = []
for mult in [0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]:
    pred, u = coverage_rerank(songs, probs, int(mult * u_win), fb)
    d = dcg(pred, truth)
    rows.append((mult, u, d))
    print(f"target={mult:.1f}x  U={u:>7,}  DCG={d:.5f}", flush=True)
print(f"\n1st place: U={u_win:,} DCG={d_win:.5f}")
# D is defined on the test set; express it as a ratio of the number of predicted slots
# (test: 715,320) and scale to the validation size: slots, played songs, meta songs.
for name, ratio in [("D = #slots", 1.0), ("D = #played songs", 773_314 / 715_320), ("D = #meta songs", 1_030_712 / 715_320)]:
    D = ratio * 5 * nq
    best = max(rows, key=lambda t: 0.8 * t[2] + 0.2 * t[1] / D)
    print(f"{name:18s}: best target {best[0]:.1f}x -> score {0.8*best[2]+0.2*best[1]/D:.5f}  "
          f"(1st: {0.8*d_win+0.2*u_win/D:.5f}, ours@1.0x: {0.8*rows[1][2]+0.2*rows[1][1]/D:.5f})")
