"""Stage 2: learn-to-rank the candidates with LightGBM and evaluate on held-out validation queries.

Validation queries are split in two halves (A: fit the ranker, B: evaluate), so the reported
score is never computed on queries the ranker has seen.
"""
import sys
import time
import numpy as np
import polars as pl
import lightgbm as lgb
from common import WORK
from features import add_features
from metric import report

t0 = time.time()
cand = pl.read_parquet(WORK / "cand_val.parquet")
truth = np.load(WORK / "val_truth.npy")
nq = len(truth)
cand = add_features(cand, "val")
lab = pl.DataFrame({"qid": np.repeat(np.arange(nq, dtype=np.int32), 5), "cand": truth.reshape(-1)}).unique()
cand = cand.join(lab.with_columns(pl.lit(1, pl.Int8).alias("y")), on=["qid", "cand"], how="left") \
           .with_columns(pl.col("y").fill_null(0)).sort("qid")
feat_cols = [c for c in cand.columns if c not in ("qid", "cand", "y")]
print("rows", cand.height, "features", len(feat_cols), "pos rate", cand["y"].mean(), f"t={time.time()-t0:.0f}s", flush=True)

rng = np.random.default_rng(1)
half = rng.random(nq) < 0.5
qid = cand["qid"].to_numpy()
mA, mB = half[qid], ~half[qid]


def top5(df, score_col):
    """Top-5 distinct songs per query; pad with the session's last songs if short."""
    s = (df.sort(["qid", score_col], descending=[False, True])
           .group_by("qid", maintain_order=True).head(5)
           .group_by("qid", maintain_order=True).agg(pl.col("cand")))
    pred = np.full((nq, 5), -1, np.int64)
    for q, lst in zip(s["qid"].to_list(), s["cand"].to_list()):
        pred[q, :len(lst)] = lst
    return pred


evalq = np.nonzero(~half)[0]
np.save(WORK / "evalq.npy", evalq)
report(np.load(WORK / "winner_pred_val.npy")[evalq], truth[evalq], tag="1st-place only_ngrams (port)")
dB = cand.filter(pl.Series(mB))
report(top5(dB, "hscore")[evalq], truth[evalq], tag="heuristic hscore (seq-rules only)")

X = cand.select(feat_cols).to_numpy().astype(np.float32)
y = cand["y"].to_numpy()
params = dict(objective="lambdarank", metric="ndcg", eval_at=[5], learning_rate=0.05,
              num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8, bagging_fraction=0.8,
              bagging_freq=1, lambdarank_truncation_level=10, verbose=-1, num_threads=12)
grpA = np.bincount(qid[mA])[np.unique(qid[mA])]
grpB = np.bincount(qid[mB])[np.unique(qid[mB])]
dtr = lgb.Dataset(X[mA], y[mA], group=grpA, feature_name=feat_cols)
dva = lgb.Dataset(X[mB], y[mB], group=grpB, reference=dtr)
model = lgb.train(params, dtr, num_boost_round=int(sys.argv[1]) if len(sys.argv) > 1 else 600,
                  valid_sets=[dva], callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50)])
model.save_model(str(WORK / "lgb_val.txt"))
cand = cand.with_columns(pl.Series("lgb", model.predict(X, num_threads=12)))
report(top5(cand.filter(pl.Series(mB)), "lgb")[evalq], truth[evalq], tag="LightGBM lambdarank")
imp = sorted(zip(model.feature_importance("gain"), feat_cols), reverse=True)[:25]
print("top features:", [(f, int(g)) for g, f in imp])
print(f"best_iter={model.best_iteration} t={time.time()-t0:.0f}s")
