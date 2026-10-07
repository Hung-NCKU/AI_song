"""Compare train (validation queries) vs test sessions: is the test set distributed like our validation?"""
import numpy as np
from common import WORK

train = np.load(WORK / "train_seq.npy")
test = np.load(WORK / "test_seq.npy")
val_q = np.load(WORK / "val_queries.npy")
ts = np.load(WORK / "train_status.npy")[:, :20]
es = np.load(WORK / "test_status.npy")
tt = np.load(WORK / "train_time.npy")[:, :20]
et = np.load(WORK / "test_time.npy")
tl = np.load(WORK / "train_login.npy")[:, :20]
el = np.load(WORK / "test_login.npy")
pop = np.bincount(np.concatenate([train.reshape(-1), test.reshape(-1)]), minlength=train.max() + 1)


def stats(name, Q, S, T, Lg):
    uniq = np.array([len(set(r)) for r in Q])
    print(f"{name:6s} n={len(Q)} last==prev={np.mean(Q[:, 19] == Q[:, 18]):.4f} uniq20={uniq.mean():.2f} "
          f"logpop(last)={np.log1p(pop[Q[:, 19]]).mean():.3f} status_mode={np.bincount((S.reshape(-1) + 9)).argmax() - 9} "
          f"status1={np.mean(S == 1):.3f} status3={np.mean(S == 3):.3f} login0={np.mean(Lg == 0):.3f} "
          f"dur_min={np.median((T[:, 19] - T[:, 0]) / 60):.1f} hour={np.median((T[:, 0] // 3600) % 24):.0f}")


stats("train", train[:, :20], ts, tt, tl)
stats("test", test, es, et, el)
