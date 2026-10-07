"""Candidate generation + features for next-5 prediction (set-based target).

The metric rewards a predicted song at rank i if it appears ANYWHERE in the 5 target songs
(order-free, confirmed by the hosts), so we model P(song in next-5 set | session prefix).

Sources (all mined over a corpus that never contains the query targets):
  * Sequential rules, offset-aware: context = last L songs ending at query position 20-j
    (L = 1..4, skip j = 0..2); a candidate gets credit for appearing d = k + j steps after that
    context in the corpus, k = 1..5.  Generalises n-gram / sequential-rule recommenders
    (Ludewig & Jannach, UMUAI 2018) and the 1st-place "position-decayed n-gram".
  * Repeat consumption: songs of the query itself (RepeatNet, AAAI 2019).
  * In-session continuation: songs following earlier occurrences of the last song.

Usage: python candidates.py val|test
"""
import sys
import time
import numpy as np
import polars as pl
from common import WORK

MODE = sys.argv[1]
LS = [1, 2, 3, 4]
JS = [0, 1, 2]
K = 5
TOPN = 30      # continuations kept per (context, offset)
KEEP = 80      # sequential-rule candidates kept per query (plus all repeat candidates)

t0 = time.time()
train = np.load(WORK / "train_seq.npy")
test = np.load(WORK / "test_seq.npy")
rng = np.random.default_rng(0)
perm = rng.permutation(len(train))
n_val = len(train) // 5
val_idx, fit_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])

if MODE == "val":
    corpus = [train[fit_idx], train[val_idx][:, :20], test]
    queries = train[val_idx][:, :20]
    truth = train[val_idx][:, 20:]
    np.save(WORK / "val_truth.npy", truth)
    np.save(WORK / "val_queries.npy", queries)
else:
    corpus = [train, test]
    queries = test
nq = len(queries)
print(MODE, "queries", nq, "corpus sessions", sum(len(c) for c in corpus), flush=True)

ccols = lambda L: [f"c{i}" for i in range(L)]


def pairs(L, d):
    """All (context of length L, song d steps later) pairs in the corpus."""
    out = []
    for A in corpus:
        T = A.shape[1]
        for i in range(L - 1, T - d):
            block = {f"c{m}": A[:, i - L + 1 + m] for m in range(L)}
            block["nxt"] = A[:, i + d]
            out.append(pl.DataFrame(block))
    return pl.concat(out)


def query_ctx(L, j):
    end = 20 - j  # exclusive
    return pl.DataFrame({"qid": np.arange(nq, dtype=np.int32),
                         **{f"c{m}": queries[:, end - L + m] for m in range(L)}})


keys = ["qid", "cand"]
feats = []  # one frame per (L, j): qid, cand, ps_L{L}j{j}, p1_L{L}j{j}, n_L{L}j{j}
ctx_tot = {}  # (L, j) -> qid, total continuation mass at d = j + 1 (context support)
for L in LS:
    qctx = pl.concat([query_ctx(L, j).drop("qid") for j in JS]).unique()
    per_j = {j: [] for j in JS}
    for d in range(1, K + max(JS) + 1):
        js = [j for j in JS if 1 <= d - j <= K]
        p = pairs(L, d).join(qctx, on=ccols(L), how="semi")
        cnt = p.group_by(ccols(L) + ["nxt"]).len("cnt")
        tot = cnt.group_by(ccols(L)).agg(pl.col("cnt").sum().alias("tot"))
        cnt = (cnt.sort("cnt", descending=True)
                  .group_by(ccols(L), maintain_order=True).head(TOPN)
                  .join(tot, on=ccols(L)))
        del p
        for j in js:
            k = d - j
            f = (query_ctx(L, j).join(cnt, on=ccols(L))
                 .select("qid", pl.col("nxt").alias("cand"), "cnt",
                         (pl.col("cnt") / pl.col("tot")).cast(pl.Float32).alias("p"),
                         pl.lit(k, pl.Int8).alias("k"), "tot"))
            per_j[j].append(f)
    for j in JS:
        f = pl.concat(per_j[j])
        sup = f.filter(pl.col("k") == 1).group_by("qid").agg(pl.col("tot").first().alias(f"sup_L{L}j{j}"))
        agg = f.group_by(keys).agg(
            pl.col("p").sum().alias(f"ps_L{L}j{j}"),
            pl.col("p").filter(pl.col("k") == 1).sum().alias(f"p1_L{L}j{j}"),
            pl.col("cnt").sum().cast(pl.Float32).alias(f"n_L{L}j{j}"))
        feats.append(agg)
        ctx_tot[(L, j)] = sup
        print(f"L={L} j={j} rows={agg.height} t={time.time() - t0:.0f}s", flush=True)
    del per_j

# Session-neighbourhood signal from EARLIER songs (L = 1, skip j = 3..9): continuations of songs
# deeper in the session, decayed by distance.  Cheap analogue of (V-)SKNN (Ludewig & Jannach 2018)
# that helps when the most recent context is rare.
FAR_J, FAR_TOP = range(3, 10), 10
far_parts = []
for d in range(min(FAR_J) + 1, max(FAR_J) + K + 1):
    js = [j for j in FAR_J if 1 <= d - j <= K]
    qctx = pl.DataFrame({"c0": np.unique(queries[:, [19 - j for j in js]])})
    p = pairs(1, d).join(qctx, on="c0", how="semi")
    cnt = p.group_by(["c0", "nxt"]).len("cnt")
    tot = cnt.group_by("c0").agg(pl.col("cnt").sum().alias("tot"))
    cnt = (cnt.sort("cnt", descending=True).group_by("c0", maintain_order=True).head(FAR_TOP)
              .join(tot, on="c0").select("c0", "nxt", (pl.col("cnt") / pl.col("tot")).alias("p")))
    del p
    for j in js:
        q = pl.DataFrame({"qid": np.arange(nq, dtype=np.int32), "c0": queries[:, 19 - j]})
        far_parts.append(q.join(cnt, on="c0").select("qid", pl.col("nxt").alias("cand"),
                                                     (pl.col("p") * 0.8 ** (j - 3)).alias("far")))
    far_parts = [pl.concat(far_parts).group_by(keys).agg(pl.col("far").sum())]
far = far_parts[0].with_columns(pl.col("far").cast(pl.Float32))
print("far rows", far.height, f"t={time.time() - t0:.0f}s", flush=True)
far_top = far.sort(["qid", "far"], descending=[False, True]).group_by("qid", maintain_order=True).head(15)

# Union of sequential-rule candidates, joined with all their features.
base = pl.concat([f.select(keys) for f in feats] + [far_top.select(keys)]).unique()
for f in feats:
    base = base.join(f, on=keys, how="left")
base = base.join(far, on=keys, how="left")
fcols = [c for c in base.columns if c not in keys]
base = base.with_columns([pl.col(c).fill_null(0) for c in fcols])
# heuristic pre-score: longer contexts are more reliable, nearer skips more relevant
h = sum(pl.col(f"ps_L{L}j{j}") * (2.0 ** L) * (0.5 ** j) for L in LS for j in JS) + 0.1 * pl.col("far")
base = (base.with_columns(h.alias("hscore"))
            .sort(["qid", "hscore"], descending=[False, True])
            .group_by("qid", maintain_order=True).head(KEEP))
print("seq-rule candidates kept:", base.height, f"t={time.time() - t0:.0f}s", flush=True)

# Repeat candidates: every distinct song of the query.
rep = pl.DataFrame({"qid": np.repeat(np.arange(nq, dtype=np.int32), 20),
                    "cand": queries.reshape(-1),
                    "pos": np.tile(np.arange(20, dtype=np.int8), nq)})
rep = rep.group_by(keys).agg(pl.col("pos").max().alias("last_pos"),
                             pl.col("pos").min().alias("first_pos"),
                             pl.len().cast(pl.Int8).alias("src_cnt"))
# In-session continuation: songs that followed earlier occurrences of the last song.
ins = []
for i in range(19):
    m = queries[:, i] == queries[:, 19]
    for k in range(1, K + 1):
        if i + k < 20 and m.any():
            ins.append(pl.DataFrame({"qid": np.nonzero(m)[0].astype(np.int32), "cand": queries[m, i + k]}))
ins = pl.concat(ins).group_by(keys).len("insess_cnt")

cand = pl.concat([base.select(keys), rep.select(keys)]).unique()
cand = (cand.join(base, on=keys, how="left").join(rep, on=keys, how="left").join(ins, on=keys, how="left"))
for (L, j), sup in ctx_tot.items():
    cand = cand.join(sup, on="qid", how="left")
num = [c for c in cand.columns if c not in keys + ["last_pos", "first_pos"]]
cand = cand.with_columns([pl.col(c).fill_null(0) for c in num] +
                         [pl.col("last_pos").fill_null(-1), pl.col("first_pos").fill_null(-1)])
print("candidates:", cand.shape, "per query:", cand.height / nq, f"t={time.time() - t0:.0f}s", flush=True)
cand.write_parquet(WORK / f"cand_{MODE}.parquet")

if MODE == "val":
    tr = pl.DataFrame({"qid": np.repeat(np.arange(nq, dtype=np.int32), K), "cand": truth.reshape(-1)}).unique()
    hit = cand.join(tr, on=keys, how="inner").height
    print(f"candidate recall (distinct target songs) = {hit / tr.height:.4f}")
