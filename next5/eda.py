"""Exploratory data analysis for KKCompany Data Game 2023 (next-5 song prediction)."""
import polars as pl
from common import DATA

pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(200)

for name in ["label_train_source", "label_train_target", "label_test_source", "meta_song",
             "meta_song_composer", "meta_song_genre", "meta_song_lyricist",
             "meta_song_producer", "meta_song_titletext"]:
    df = pl.read_parquet(DATA / f"{name}.parquet")
    print(f"\n===== {name} {df.shape}")
    print(df.head(3))
    print(df.null_count())

src = pl.read_parquet(DATA / "label_train_source.parquet")
tgt = pl.read_parquet(DATA / "label_train_target.parquet")
tst = pl.read_parquet(DATA / "label_test_source.parquet")
sample = pl.read_csv(DATA / "sample.csv")
print("\nsample.csv", sample.shape)
print(sample.head(3))

print("\n#sessions train/test:", src["session_id"].n_unique(), tst["session_id"].n_unique())
print("songs per session (train src):", src.group_by("session_id").len()["len"].describe())
print("songs per session (train tgt):", tgt.group_by("session_id").len()["len"].describe())
print("session id overlap train/test:",
      len(set(src["session_id"].unique()) & set(tst["session_id"].unique())))
print("listening_order range src/tgt/test:",
      src["listening_order"].min(), src["listening_order"].max(),
      tgt["listening_order"].min(), tgt["listening_order"].max(),
      tst["listening_order"].min(), tst["listening_order"].max())

# How often do target songs repeat songs already in the source?
s = src.group_by("session_id").agg(pl.col("song_id").alias("src"))
t = tgt.sort(["session_id", "listening_order"]).group_by("session_id", maintain_order=True).agg(pl.col("song_id").alias("tgt"))
st = s.join(t, on="session_id")
rows = st.to_dicts()
n = len(rows)
rep = sum(sum(x in set(r["src"]) for x in r["tgt"]) for r in rows) / (5 * n)
any_rep = sum(any(x in set(r["src"]) for x in r["tgt"]) for r in rows) / n
last5_pos = [sum(r["tgt"][k] == r["src"][15 + k] for r in rows) / n for k in range(5)]
print(f"\nfrac target songs seen in source: {rep:.4f}; sessions with any repeat: {any_rep:.4f}")
print("P(tgt[k] == src[15+k]) (playlist loop):", [round(x, 4) for x in last5_pos])
print("P(tgt[0] == src[-1]):", sum(r["tgt"][0] == r["src"][-1] for r in rows) / n)
uniq_tgt = sum(len(set(r["tgt"])) for r in rows) / n
print("avg unique songs in target:", uniq_tgt)

# Song popularity
all_songs = pl.concat([src.select("song_id"), tgt.select("song_id"), tst.select("song_id")])
print("\nunique songs overall:", all_songs["song_id"].n_unique())
tgt_songs = set(tgt["song_id"].unique())
train_src_songs = set(src["song_id"].unique())
print("target songs never in any train source:", len(tgt_songs - train_src_songs) / len(tgt_songs))
test_songs = set(tst["song_id"].unique())
print("test source songs unseen in train:", len(test_songs - train_src_songs - tgt_songs) / len(test_songs))

# time info
print("\nplayed_at range:", src["unix_played_at"].min(), src["unix_played_at"].max(),
      tst["unix_played_at"].min(), tst["unix_played_at"].max())
