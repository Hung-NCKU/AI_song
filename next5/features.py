"""Extra candidate features: popularity in the corpus and metadata agreement with the session."""
import numpy as np
import polars as pl
from common import WORK


def corpus_for(mode, frac=1.0):
    train = np.load(WORK / "train_seq.npy")
    test = np.load(WORK / "test_seq.npy")
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(train))
    n_val = len(train) // 5
    val_idx, fit_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    if mode == "val":
        if frac < 1.0:  # shrink the fit corpus to measure the effect of corpus size
            fit_idx = fit_idx[np.random.default_rng(2).random(len(fit_idx)) < frac]
        return [train[fit_idx], train[val_idx][:, :20], test], train[val_idx][:, :20]
    return [train, test], test


def add_features(cand, mode):
    corpus, queries = corpus_for(mode)
    n_songs = pl.read_parquet(WORK / "songs.parquet").height
    pop = np.zeros(n_songs, np.float32)
    for A in corpus:
        np.add.at(pop, A.reshape(-1), 1)
    meta = pl.read_parquet(WORK / "songs.parquet")
    art = meta["artist_id"].to_numpy()
    alb = meta["album_id"].to_numpy()
    lang = meta["language_id"].to_numpy()
    genre = meta["genre_id"].to_numpy()
    length = meta["song_length"].fill_null(0).to_numpy()
    month = meta["album_month"].to_numpy()

    q = cand["qid"].to_numpy()
    c = cand["cand"].to_numpy()
    last = queries[q, 19]
    qa = art[queries]          # (nq, 20) artists of each session
    qal = alb[queries]
    same_art_cnt = (qa[q] == art[c][:, None]).sum(1)
    same_alb_cnt = (qal[q] == alb[c][:, None]).sum(1)
    # stacking: rank of the candidate in the 1st-place n-gram cascade (0 = not predicted)
    wp = np.load(WORK / f"winner_pred_{mode}.npy")
    w_rank = np.zeros(len(c), np.int8)
    for r in range(5):
        w_rank[(wp[q, r] == c) & (w_rank == 0)] = r + 1
    n_uniq = np.array([len(set(r)) for r in queries], np.int8)
    n_uniq_art = np.array([len(set(r)) for r in qa], np.int8)
    return cand.with_columns(
        pl.Series("pop", np.log1p(pop[c])),
        pl.Series("last_pop", np.log1p(pop[last])),
        pl.Series("same_art_last", (art[c] == art[last]) & (art[c] >= 0)),
        pl.Series("same_alb_last", (alb[c] == alb[last]) & (alb[c] >= 0)),
        pl.Series("same_lang_last", lang[c] == lang[last]),
        pl.Series("same_genre_last", genre[c] == genre[last]),
        pl.Series("same_art_cnt", same_art_cnt.astype(np.int8)),
        pl.Series("same_alb_cnt", same_alb_cnt.astype(np.int8)),
        pl.Series("cand_len", length[c].astype(np.float32)),
        pl.Series("cand_month", month[c]),
        pl.Series("q_uniq", n_uniq[q]),
        pl.Series("q_uniq_art", n_uniq_art[q]),
        pl.Series("is_last", c == last),
        pl.Series("winner_rank", w_rank),
    )
