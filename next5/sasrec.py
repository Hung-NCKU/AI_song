"""SASRec (Kang & McAuley, ICDM 2018) for next-5 prediction, as a standalone model and as an extra
candidate source / feature for the LightGBM ranker.

Adaptations to this task:
  * Multi-offset objective: at every position t the model predicts each of the songs at t+1..t+5
    (the metric is set-based over the next 5 songs).
  * Sampled softmax over the 770k-song catalogue with 16,384 shared negatives drawn from
    unigram^0.75 and logQ correction -- an approximation of the full cross-entropy that makes SASRec
    strong (Klenitskiy & Vasilev, RecSys 2023; Petrov & Macdonald, RecSys 2023).
  * Trained once on the validation-mode corpus (other train sessions' 25 songs + val/test prefixes),
    which contains no validation target, so validation numbers are honest; the same model scores
    the test sessions.

Usage: python sasrec.py            # train, evaluate standalone on validation, write features
Outputs: work/sasrec.pt, work/sas_val.parquet, work/sas_test.parquet
"""
import math
import os
import sys
import time
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from common import WORK
from features import corpus_for
from metric import report

torch.manual_seed(0)
np.random.seed(0)
DEV = "cuda"
D, LAYERS, HEADS, DROP = 128, 2, 4, 0.2
T = 25
BATCH, NNEG, EPOCHS, LR = 512, 8192, 6, 2e-3
TOPK = 50
SMOKE = os.environ.get("SMOKE") == "1"  # quick functional test
if SMOKE:
    EPOCHS = 1
t0 = time.time()

corpus, val_q = corpus_for("val")
test_q = np.load(WORK / "test_seq.npy")
truth = np.load(WORK / "val_truth.npy")
n_items = pl.read_parquet(WORK / "songs.parquet").height
cnt = np.zeros(n_items, np.float64)
for A in corpus:
    np.add.at(cnt, A.reshape(-1), 1)
known = cnt > 0  # songs never seen in the corpus have untrained embeddings -> never recommended


def pad(A):  # item ids shifted by +1 (0 = padding), right-padded to length T
    out = np.zeros((len(A), T), np.int64)
    out[:, :A.shape[1]] = A + 1
    return out


seqs = torch.from_numpy(np.concatenate([pad(A) for A in corpus]))
if SMOKE:
    seqs = seqs[torch.randperm(len(seqs))[:60000]]
print("training sequences", tuple(seqs.shape), "known items", int(known.sum()), flush=True)
q = cnt ** 0.75
q /= q.sum()
q_t = torch.tensor(np.concatenate([[1.0], q]), dtype=torch.float32, device=DEV)  # index 0 unused
log_q = torch.log(q_t.clamp_min(1e-12))


class SASRec(nn.Module):
    def __init__(self):
        super().__init__()
        self.item = nn.Embedding(n_items + 1, D, padding_idx=0)
        self.pos = nn.Embedding(T, D)
        layer = nn.TransformerEncoderLayer(D, HEADS, 4 * D, DROP, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, LAYERS, enable_nested_tensor=False)
        self.ln = nn.LayerNorm(D)
        self.drop = nn.Dropout(DROP)
        nn.init.normal_(self.item.weight, std=0.02)
        self.register_buffer("causal", torch.triu(torch.ones(T, T, dtype=torch.bool), 1))

    def forward(self, x):
        L = x.shape[1]
        h = self.item(x) * math.sqrt(D) + self.pos.weight[:L]
        h = self.enc(self.drop(h), mask=self.causal[:L, :L], is_causal=True)
        return self.ln(h)


model = SASRec().to(DEV)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)
steps = EPOCHS * math.ceil(len(seqs) / BATCH)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
eval_idx = np.random.default_rng(3).choice(len(val_q), 20000, replace=False)


@torch.no_grad()
def user_vecs(Q, bs=4096):
    model.eval()
    out = []
    for i in range(0, len(Q), bs):
        x = torch.from_numpy(Q[i:i + bs] + 1).to(DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(model(x)[:, -1].float())
    model.train()
    return torch.cat(out)


@torch.no_grad()
def topk_items(U, k=TOPK, bs=256):
    E = model.item.weight[1:].float()
    mask = torch.from_numpy(~known).to(DEV)
    S, I = [], []
    for i in range(0, len(U), bs):
        s = U[i:i + bs] @ E.T
        s[:, mask] = -1e9
        v, ix = s.topk(k, dim=1)
        S.append(v.cpu())
        I.append(ix.cpu())
    return torch.cat(S).numpy(), torch.cat(I).numpy()


def evaluate(tag, idx=None):
    Q = val_q if idx is None else val_q[idx]
    tr = truth if idx is None else truth[idx]
    _, I = topk_items(user_vecs(Q), 5)
    return report(I, tr, tag=tag)[0]


step = 0
for ep in range(EPOCHS):
    perm = torch.randperm(len(seqs))
    tot, n = 0.0, 0
    for b in range(0, len(seqs), BATCH):
        x = seqs[perm[b:b + BATCH]].to(DEV, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h = model(x)                                         # (B, T, D)
            neg = torch.multinomial(q_t, NNEG, replacement=True)
            En = model.item(neg)                                 # (N, D)
            neg_logit = (h @ En.T).float() - (log_q[neg] + math.log(NNEG))
            lse_neg = torch.logsumexp(neg_logit, dim=-1)         # (B, T)
            loss_sum, m_sum = 0.0, 0.0
            for k in range(1, 6):
                tgt = x[:, k:]                                   # songs k steps later
                m = (tgt > 0) & (x[:, :-k] > 0)
                pos = (h[:, :-k] * model.item(tgt)).sum(-1).float() - (log_q[tgt] + math.log(NNEG))
                l = torch.logaddexp(pos, lse_neg[:, :-k]) - pos
                loss_sum = loss_sum + (l * m).sum()
                m_sum = m_sum + m.sum()
            loss = loss_sum / m_sum
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        tot += loss.item()
        n += 1
        step += 1
    print(f"epoch {ep + 1} loss={tot / n:.4f} t={time.time() - t0:.0f}s", flush=True)
    evaluate(f"SASRec ep{ep + 1} (20k val)", eval_idx)

if SMOKE:
    print(f"smoke ok, peak GPU mem {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB"); sys.exit()
torch.save(model.state_dict(), WORK / "sasrec.pt")
half = np.random.default_rng(1).random(len(val_q)) < 0.5
d_all = evaluate("SASRec standalone, all val")
evaluate("SASRec standalone, val half B", np.nonzero(~half)[0])


def features(Q, cand_path, out_path):
    """SASRec top-K candidates + SASRec score for every existing ranker candidate."""
    U = user_vecs(Q)
    S, I = topk_items(U)
    top1 = S[:, 0]
    nq = len(Q)
    top = pl.DataFrame({"qid": np.repeat(np.arange(nq, dtype=np.int32), TOPK), "cand": I.reshape(-1).astype(np.int32),
                        "sas_rank": np.tile(np.arange(1, TOPK + 1, dtype=np.int16), nq)})
    pairs = pl.concat([pl.read_parquet(cand_path).select("qid", "cand"), top.select("qid", "cand")]).unique()
    qi = torch.from_numpy(pairs["qid"].to_numpy().astype(np.int64))
    ci = torch.from_numpy(pairs["cand"].to_numpy().astype(np.int64))
    E = model.item.weight[1:].float()
    sc = []
    with torch.no_grad():
        for i in range(0, len(qi), 1_000_000):
            a, c = qi[i:i + 1_000_000].to(DEV), ci[i:i + 1_000_000].to(DEV)
            sc.append((U[a] * E[c]).sum(-1).cpu())
    score = torch.cat(sc).numpy()
    out = pairs.with_columns(pl.Series("sas_score", score.astype(np.float32)),
                             pl.Series("sas_rel", (score - top1[pairs["qid"].to_numpy()]).astype(np.float32)))
    out = out.join(top, on=["qid", "cand"], how="left").with_columns(pl.col("sas_rank").fill_null(TOPK + 1))
    out.write_parquet(out_path)
    print(out_path.name, out.shape, flush=True)


features(val_q, WORK / "cand_val.parquet", WORK / "sas_val.parquet")
features(test_q, WORK / "cand_test.parquet", WORK / "sas_test.parquet")
if True:  # candidate recall of SASRec top-K alone and added to the rule-based pool
    sv = pl.read_parquet(WORK / "sas_val.parquet")
    lab = pl.DataFrame({"qid": np.repeat(np.arange(len(truth), dtype=np.int32), 5), "cand": truth.reshape(-1).astype(np.int32)}).unique()
    base = pl.read_parquet(WORK / "cand_val.parquet").select("qid", "cand")
    r_sas = sv.filter(pl.col("sas_rank") <= TOPK).join(lab, on=["qid", "cand"]).height / lab.height
    r_union = pl.concat([base, sv.filter(pl.col("sas_rank") <= 30).select("qid", "cand")]).unique().join(lab, on=["qid", "cand"]).height / lab.height
    print(f"recall: SASRec top{TOPK}={r_sas:.4f}  rules+SASRec top30={r_union:.4f}  t={time.time() - t0:.0f}s")
