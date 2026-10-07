"""Is the model's self-estimated DCG (sum of top-5 probabilities x weights) close to the realised DCG on
validation?  Justifies reading the test self-estimate as a DCG estimate."""
import numpy as np
import polars as pl
from common import WORK
from metric import W, dcg
from postproc import topk
from legacy_proxy import pred as legacy_pred  # noqa: E402  (prints its own report)

oof = pl.read_parquet(WORK / "val_oof.parquet")
truth = np.load(WORK / "val_truth.npy")
songs, probs = topk(oof, len(truth), 5)
print(f"val: self-estimated DCG={np.mean(probs @ W):.4f}  realised DCG of the same top-5={dcg(songs, truth):.4f}")
print(f"legacy proxy distinct songs U={len(np.unique(legacy_pred[legacy_pred >= 0]))}")
