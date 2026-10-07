"""Final pipeline: binary LightGBM ranker with 2-fold CV on validation queries, coverage-aware
re-ranking (postproc.py), and the test-set submission.

Since our DCG is higher and our distinct-song count is >= the 1st place's, our score is higher
for ANY Coverage denominator D (the official scorer is offline, so D cannot be calibrated).

Usage: python train_final.py            # CV report + test submission
       python train_final.py val_only   # CV report only
       python train_final.py test_only  # test submission only
"""
import sys
import time
import numpy as np
import polars as pl
import lightgbm as lgb
from common import DATA, WORK
from features import add_features
from metric import report
from postproc import topk, coverage_rerank, fallback_for

t0 = time.time()
MODE = sys.argv[1] if len(sys.argv) > 1 else "all"
USE_SAS = "sas" in sys.argv[2:]  # hybrid with SASRec (sasrec.py must have been run)
TAG = "_sas" if USE_SAS else ""
PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, num_threads=12)
ROUNDS = 400
ALT = 25  # candidates kept per session for swaps


def load(mode):
    cand = pl.read_parquet(WORK / f"cand_{mode}.parquet")
    if USE_SAS:  # hybrid: SASRec top-30 as extra candidates + SASRec scores as features
        sas = pl.read_parquet(WORK / f"sas_{mode}.parquet").with_columns(pl.col("cand").cast(cand["cand"].dtype),
                                                                          pl.col("qid").cast(cand["qid"].dtype))
        new = sas.filter(pl.col("sas_rank") <= 30).select("qid", "cand")
        cand = pl.concat([cand, new.join(cand.select("qid", "cand"), on=["qid", "cand"], how="anti")], how="diagonal")
        cand = cand.join(sas, on=["qid", "cand"], how="left")
        cand = cand.with_columns([pl.col(c).fill_null(-1 if c in ("last_pos", "first_pos") else 0)
                                  for c in cand.columns if c not in ("qid", "cand")])
    cand = add_features(cand, mode)
    if mode == "val":
        truth = np.load(WORK / "val_truth.npy")
        lab = pl.DataFrame({"qid": np.repeat(np.arange(len(truth), dtype=np.int32), 5),
                            "cand": truth.reshape(-1)}).unique().with_columns(pl.lit(1, pl.Int8).alias("y"))
        cand = cand.join(lab, on=["qid", "cand"], how="left").with_columns(pl.col("y").fill_null(0))
    return cand.sort("qid")


def with_popular(fallback, popular):
    """Session's recent songs first, then globally popular songs (never a dummy)."""
    return [[x for x in f if x >= 0] + [p for p in popular if p not in f] for f in fallback]


popular = np.bincount(np.load(WORK / "train_seq.npy").reshape(-1)).argsort()[::-1][:20].tolist()
val = load("val")
feat_cols = [c for c in val.columns if c not in ("qid", "cand", "y")]
X = val.select(feat_cols).to_numpy().astype(np.float32)
y = val["y"].to_numpy()

if MODE in ("all", "val_only"):
    truth = np.load(WORK / "val_truth.npy")
    queries = np.load(WORK / "val_queries.npy")
    nq = len(truth)
    qid = val["qid"].to_numpy()
    fold = (np.random.default_rng(1).random(nq) < 0.5).astype(int)  # same halves as rank.py
    oof = np.zeros(len(val))
    for f in (0, 1):
        tr = fold[qid] != f
        m = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr], feature_name=feat_cols), num_boost_round=ROUNDS)
        oof[~tr] = m.predict(X[~tr], num_threads=12)
        print(f"fold {f} done t={time.time()-t0:.0f}s", flush=True)
    oofdf = val.select("qid", "cand", "y").with_columns(pl.Series("p", oof))
    oofdf.write_parquet(WORK / f"val_oof{TAG}.parquet")
    songs, probs = topk(oofdf, nq, ALT)
    fb = with_popular(fallback_for(queries), popular)
    win = np.load(WORK / "winner_pred_val.npy")
    u_win = len(np.unique(win))
    d_w, _ = report(win, truth, tag="1st-place only_ngrams (port), all val")
    pred0, u0 = coverage_rerank(songs, probs, 0, fb)
    d0, _ = report(pred0, truth, tag="ours: LightGBM top-5 (no coverage control)")
    pred1, u1 = coverage_rerank(songs, probs, u_win, fb)
    d1, _ = report(pred1, truth, tag="ours: + coverage control (U >= 1st place)")
    print(f"distinct songs U: 1st={u_win} ours_raw={u0} ours_cov={u1}")
    for D in [715_320, 773_314, 1_030_712]:
        Dv = D * nq / 143_064  # same ratio to the number of sessions as on test
        print(f"D_test={D:>9,}: 1st={0.8*d_w+0.2*u_win/Dv:.5f}  ours_raw={0.8*d0+0.2*u0/Dv:.5f}  ours_cov={0.8*d1+0.2*u1/Dv:.5f}")
    np.save(WORK / f"ours_pred_val{TAG}.npy", pred1)

if MODE in ("all", "test_only"):
    model = lgb.train(PARAMS, lgb.Dataset(X, y, feature_name=feat_cols), num_boost_round=ROUNDS)
    model.save_model(str(WORK / f"lgb_final{TAG}.txt"))
    del X, val
    test = load("test")
    tq = np.load(WORK / "test_seq.npy")
    test = test.with_columns(pl.Series("p", model.predict(test.select(feat_cols).to_numpy().astype(np.float32), num_threads=12)))
    test.select("qid", "cand", "p").write_parquet(WORK / f"test_pred{TAG}.parquet")
    songs, probs = topk(test, len(tq), ALT)
    u_win_test = len(np.unique(np.load(WORK / "winner_pred_test.npy")))
    pred, u = coverage_rerank(songs, probs, u_win_test, with_popular(fallback_for(tq), popular))
    assert (pred >= 0).all() and all(len(set(r)) == 5 for r in pred)
    exp_dcg = float(np.mean(np.sort(probs[:, :5])[:, ::-1] @ np.array([1, .63, .5, .43, .38])))
    print(f"test: distinct U ours={u} 1st={u_win_test}; model-expected DCG of raw top-5={exp_dcg:.4f}")
    np.save(WORK / f"ours_pred_test{TAG}.npy", pred)
    song_ids = pl.read_parquet(WORK / "songs.parquet")["song_id"].to_numpy()
    sub = pl.DataFrame({"session_id": np.load(WORK / "test_sid.npy"), **{f"top{i+1}": song_ids[pred[:, i]] for i in range(5)}})
    out = pl.read_csv(DATA / "sample.csv").select("session_id").join(sub, on="session_id", how="left")
    assert out.null_count().sum_horizontal().item() == 0
    out.write_csv(WORK / f"submission{TAG}.csv")
    print(f"submission written: {out.shape} t={time.time()-t0:.0f}s")
