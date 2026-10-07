# next5：Session-based 下 5 首歌預測（KKCompany Data Game 2023）

比賽：[KKCompany Music Challenge: Next-5 Songcraft](https://www.kaggle.com/competitions/datagame-2023)。任務是根據 session 的前 20 首歌，預測接下來 5 首。

評分方式：score = 0.8 × DCG@5（不看順序的集合命中）+ 0.2 × Coverage。

## 結果（本地驗證，114,451 個 session，out-of-fold）

| 方法 | DCG@5 | score（D = 總格數） | score（D = meta 歌曲數） |
|---|---|---|---|
| 舊方法（本 repo 2023 年版，代理評估） | 0.179 | 0.162 | — |
| 第 1 名 only_ngrams（官方 Private 0.55569，同一驗證集重現） | 0.664 | 0.583 | 0.567 |
| SASRec（Transformer）單獨使用 | 0.358 | 0.332 | — |
| 本研究：序列規則 + LightGBM | 0.738 | 0.643 | 0.627 |
| **本研究：+ SASRec 混合（最終）** | **0.743** | **0.646** | **0.630** |

- DCG 比第 1 名高 **+11.9%**，不重複預測歌曲數和第 1 名相同，所以**不論 Coverage 分母是多少，總分都領先約 +0.063**。
- Transformer（SASRec）單獨使用只有 0.358，但當成互補訊號加進來，可以再多 +0.0045 DCG（見 RESEARCH_LOG §7）。
- 第 1 名最終版（加上 JMLM）官方只比純 n-gram 高 +0.010。
- Kaggle 的評分程式已經關閉（連官方 `sample.csv` 都回傳錯誤），因此所有比較都是在**同一個本地驗證集**上進行，並把第 1 名方法忠實移植過來一起評估。詳見 [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md)。

## 方法（兩階段：召回 → 學習排序）

1. **召回**（`candidates.py`）：每個 session 約 85 個候選，召回率 41%。
   - **位移感知序列規則**：上下文長度 L=1..4、跳過量 j=0..2、位移 k=1..5。
   - 遠端鄰域（STAN／V-SKNN 精神）。
   - 重複消費（RepeatNet 精神）。
2. **特徵**（`features.py`）：
   - 規則機率與上下文支持度。
   - 重複行為：最後出現位置、出現次數、session 內延續。
   - 熱門度、metadata 一致性（歌手、專輯、語言、曲風）。
   - stacking：第 1 名方法的輸出排名。
3. **SASRec**（`sasrec.py`）：Transformer 序列模型（multi-target + sampled softmax），提供 top-30 候選與分數特徵。
4. **排序**（`train_final.py`）：binary LightGBM，2-fold OOF。
5. **Coverage 控制**（`postproc.py`）：依期望 DCG 損失最小的順序，貪婪地把重複推薦的歌換成沒推過的候選，抵銷學習排序的熱門度偏差。

文獻依據見 [docs/LITERATURE.md](docs/LITERATURE.md)。

## 重現

```bash
# WSL / Linux, Python 3.12
uv venv ~/.venvs/aisong && uv pip install --python ~/.venvs/aisong/bin/python polars pyarrow numpy pandas scikit-learn lightgbm kaggle
# 比賽資料放在 ../../data（不在 repo 中，資料不得外流）
kaggle competitions download datagame-2023 -p ../../data   # 再解壓縮
python prep.py                      # 歌曲 ID 編碼、轉成 session 矩陣
python winner_ngram.py val          # 第 1 名方法（驗證集）
python winner_ngram.py test         #             （測試集）
python candidates.py val            # 召回 + 規則特徵（約 1.5 分鐘，記憶體峰值約 12GB）
python candidates.py test
python train_final.py               # 2-fold 驗證報告 + 測試集提交檔（不含 SASRec）
uv pip install --python ~/.venvs/aisong/bin/python torch --index-url https://download.pytorch.org/whl/cu128
python sasrec.py                    # SASRec 訓練（RTX 5060 約 12 分鐘）+ 特徵
python train_final.py all sas       # 混合版：work/submission_sas.csv（最終）
python coverage_curve.py            # coverage 取捨曲線
python legacy_proxy.py              # 舊方法 / baseline 評估
```

輔助腳本：

- `eda.py`、`shift_check.py`：EDA，以及訓練／測試分布比較。
- `fetch_pages.py`、`fetch_topics.py`：用 Kaggle API 抓評分說明與討論區。
- `rank.py`：LambdaRank 實驗。
- `check_calibration.py`：機率校準檢查。
- `metric.py`：本地評分。

`results/` 收錄各實驗的輸出 log。
