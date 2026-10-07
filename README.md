# AI_song：KKBOX 下 5 首歌預測

KKCompany Data Game 2023（[datagame-2023](https://www.kaggle.com/competitions/datagame-2023)）：根據使用者在一個聆聽 session 的前 20 首歌，預測接下來的 5 首。

| 版本 | 位置 | 方法 | 分數 |
|---|---|---|---|
| 2023 課程版 | 根目錄（`model.py`、`score.py` …） | 熱門度回歸 + XGBoost | 官方 Public 0.168～0.179 |
| 2026 研究版 | [`next5/`](next5/) | 序列規則召回 + LightGBM 學習排序 + coverage 控制 | 本地驗證 0.643（同驗證集第 1 名 0.583） |

## 新舊方法比較

### 2023 年做法在做什麼

- `score.py`：把每首歌在所有資料中的出現比例當作 `score`（熱門度）。
- `model.py`：
  - 用 XGBoost 拿每筆播放的欄位（session_id、unix_played_at、artist_id …）回歸這個熱門度。
  - 對每個測試 session，從**它自己已經播過的 20 首**裡挑預測分數最高的 5 首。
- `LSTM.py`：只是用 sin 波做的 LSTM 範例，沒有接上比賽資料。

### 為什麼分數低

1. **目標錯誤**：回歸的目標是「熱門度」，不是「接下來會不會聽」。模型學到的本質上是一張熱門度表，用不到序列資訊。
2. **候選錯誤**：只從已經聽過的 20 首裡挑。但 83% 的目標歌不在前 20 首裡，所以 DCG 上限很低。
3. **違反題目精神**：比賽明確要求避免集中在熱門歌，Coverage 也佔 20%。這個做法剛好反過來推最熱門的歌，不重複預測數只有新方法的 36%。

在同一個驗證集上評估這個做法的代理版本，DCG 是 0.179、score 0.162，和當年的官方分數 0.168～0.179 一致。

### 2026 年做法改了什麼

| 面向 | 2023 | 2026 |
|---|---|---|
| 問題定義 | 對每筆播放回歸熱門度 | 預測「歌曲 ∈ 接下來 5 首」的機率（對應不看順序的 DCG） |
| 使用的訊號 | 單筆播放的欄位 | session 內的播放順序（n-gram／序列規則）、重複行為、歌曲 metadata |
| 候選 | 只有 session 內的 20 首 | session 內的歌 + 語料中「此上下文之後出現過的歌」（約 85 個／session） |
| 模型 | XGBoost 回歸 | 兩階段：召回 + LightGBM 學習排序 |
| 驗證 | 隨機切分播放紀錄、看 MSE（和評分無關） | 依比賽評分公式在獨立 session 上計算 DCG／Coverage |
| 熱門度 | 只推熱門歌 | 顯式控制 coverage |
| DCG@5（同一驗證集） | 0.179 | **0.738** |

完整的研究過程、評分公式考證、文獻調研與實驗紀錄：

- [next5/README.md](next5/README.md)
- [next5/docs/RESEARCH_LOG.md](next5/docs/RESEARCH_LOG.md)
- [next5/docs/LITERATURE.md](next5/docs/LITERATURE.md)

> 比賽資料不得外流，所以 repo 不包含資料，`.gitignore` 也排除了所有由資料產生的中間檔。
