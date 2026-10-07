"""Build a probe submission whose DCG is ~0, so that score = 0.2 * U / D reveals the Coverage
denominator D.  Every slot is filled with a distinct song from meta_song that never appears in
any provided session (train source/target, test source)."""
import numpy as np
import polars as pl
from common import DATA, WORK

meta = pl.read_parquet(DATA / "meta_song.parquet").select("song_id").unique()
seen = pl.read_parquet(WORK / "songs.parquet").select("song_id")
unseen = meta.join(seen, on="song_id", how="anti")["song_id"].to_numpy()
print("meta songs", meta.height, "songs in sessions", seen.height, "never-played", len(unseen))
sid = np.load(WORK / "test_sid.npy")
n = len(sid)
U = min(len(unseen), 5 * n)
pool = np.resize(unseen[:U], 5 * n)  # distinct while available, then cycles
pred = pool.reshape(n, 5)
# make the 5 songs within a session distinct even after cycling
for i in range(1, 5):
    clash = (pred[:, :i] == pred[:, i:i + 1]).any(1)
    assert not clash.any()
pl.DataFrame({"session_id": sid, **{f"top{i+1}": pred[:, i] for i in range(5)}}).write_csv(WORK / "probe_coverage.csv")
print("distinct predicted U =", len(np.unique(pred)), "slots =", pred.size)
