"""Competition metric (Kaggle Evaluation page + host clarifications in discussions 455783,
456720, 455784, 458283).

score = 0.8 * DCG@5 + 0.2 * Coverage            (max = 2.94*0.8 + 1*0.2 = 2.552)
  DCG@5    : order-free membership gain. Predictions are first de-duplicated (repeats -> dummy);
             gain_i = 1 if pred_i is in the set of the 5 target songs; weights
             [1, 0.63, 0.5, 0.43, 0.38]; averaged over sessions.
             e.g. truth [a,b,c,d,e] vs pred [e,d,c,b,a] -> full score; truth AAAAA vs ABCDE -> 1.
  Coverage : (# distinct predicted songs) / (Total song #).  The denominator is not specified;
             several candidates are reported.
"""
import numpy as np

W = np.array([1, 0.63, 0.5, 0.43, 0.38])


def dcg_per_session(pred, actual):
    pred = np.asarray(pred)
    gain = np.zeros(pred.shape, dtype=np.float64)
    for i in range(5):
        dup = np.zeros(len(pred), bool)
        for m in range(i):
            dup |= pred[:, i] == pred[:, m]
        hit = np.zeros(len(pred), bool)
        for m in range(5):
            hit |= pred[:, i] == actual[:, m]
        gain[:, i] = hit & ~dup
    return (gain * W).sum(1)


def dcg(pred, actual):
    return float(dcg_per_session(pred, actual).mean())


def coverage(pred, actual=None, n_meta_songs=None):
    n_pred = len(np.unique(pred))
    out = {"cov_slots": n_pred / pred.size}
    if actual is not None:
        out["cov_tgt"] = n_pred / len(np.unique(actual))
    if n_meta_songs is not None:
        out["cov_meta"] = n_pred / n_meta_songs
    return out


def report(pred, actual, n_meta_songs=None, tag=""):
    d = dcg(pred, actual)
    c = coverage(pred, actual, n_meta_songs)
    msg = f"[{tag}] DCG={d:.5f} " + " ".join(f"{k}={v:.4f}" for k, v in c.items())
    msg += " | score(cov_slots)=" + f"{0.8 * d + 0.2 * c['cov_slots']:.5f}"
    print(msg, flush=True)
    return d, c
