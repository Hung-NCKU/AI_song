"""Encode song ids to integers and store sessions as dense numpy matrices.

Outputs (work/):
  train_seq.npy  (n_train, 25) int32 : listening_order 1..25 (20 source + 5 target)
  test_seq.npy   (n_test, 20)  int32
  train_sid.npy / test_sid.npy       : session ids (row order)
  songs.parquet                      : song_idx -> song_id + metadata (artist/album/... as ints)
"""
import numpy as np
import polars as pl
from common import DATA, WORK

src = pl.read_parquet(DATA / "label_train_source.parquet")
tgt = pl.read_parquet(DATA / "label_train_target.parquet")
tst = pl.read_parquet(DATA / "label_test_source.parquet")

songs = pl.concat([src["song_id"], tgt["song_id"], tst["song_id"]]).unique().sort()
songs = songs.to_frame().with_row_index("song_idx").with_columns(pl.col("song_idx").cast(pl.Int32))
print("songs:", songs.height)


def to_matrix(df, length):
    df = (df.join(songs, on="song_id")
            .sort(["session_id", "listening_order"]))
    sid = df["session_id"].unique(maintain_order=True).to_numpy()
    mat = df["song_idx"].to_numpy().reshape(-1, length)
    assert len(sid) == mat.shape[0]
    return sid, mat.astype(np.int32)


train_sid, train_seq = to_matrix(pl.concat([src, tgt]), 25)
test_sid, test_seq = to_matrix(tst, 20)
np.save(WORK / "train_seq.npy", train_seq)
np.save(WORK / "test_seq.npy", test_seq)
np.save(WORK / "train_sid.npy", train_sid)
np.save(WORK / "test_sid.npy", test_sid)

# Per-play side information (same layout as the sequences) for later features.
for name, df, length in [("train", pl.concat([src, tgt]), 25), ("test", tst, 20)]:
    df = df.sort(["session_id", "listening_order"])
    np.save(WORK / f"{name}_status.npy", df["play_status"].fill_null(-9).to_numpy().reshape(-1, length).astype(np.int8))
    np.save(WORK / f"{name}_time.npy", df["unix_played_at"].to_numpy().reshape(-1, length).astype(np.int64))
    np.save(WORK / f"{name}_login.npy", df["login_type"].fill_null(-9).to_numpy().reshape(-1, length).astype(np.int8))

# Song metadata: map every hashed id column to a dense integer (-1 = missing).
meta = pl.read_parquet(DATA / "meta_song.parquet")
meta = songs.join(meta, on="song_id", how="left")
for extra, col in [("meta_song_genre", "genre_id"), ("meta_song_composer", "composer_id"),
                   ("meta_song_lyricist", "lyricist_id"), ("meta_song_producer", "producer_id")]:
    e = pl.read_parquet(DATA / f"{extra}.parquet").unique("song_id", keep="first")
    meta = meta.join(e, on="song_id", how="left")
for col in ["artist_id", "album_id", "genre_id", "composer_id", "lyricist_id", "producer_id", "album_month"]:
    meta = meta.with_columns(pl.col(col).cast(pl.Utf8).cast(pl.Categorical).to_physical().cast(pl.Int32).fill_null(-1).alias(col))
meta = meta.with_columns(pl.col("language_id").cast(pl.Int32).fill_null(-1),
                         pl.col("song_length").cast(pl.Float32))
meta.sort("song_idx").write_parquet(WORK / "songs.parquet")
print(meta.head())
