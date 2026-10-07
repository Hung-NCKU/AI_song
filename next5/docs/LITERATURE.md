# 文獻調研：Session-based 音樂推薦（Next-5 Songcraft）

本題的形式是：給定一個聆聽 session 的前 20 首歌，預測接下來 5 首「會出現哪些歌」（評分不看順序，見 [RESEARCH_LOG.md](RESEARCH_LOG.md#2-評分公式)）。這在文獻中屬於 **session-based / sequential recommendation**，並帶有 **重複消費（repeat consumption）** 與 **熱門度偏差（popularity bias）** 兩個特性。以下依「我們實際採用了什麼」整理。

## 1. 鄰域與序列規則方法（本研究的召回主幹）

| 論文 | 出處 | 重點 | 本研究如何使用 |
|---|---|---|---|
| Ludewig & Jannach, *Evaluation of session-based recommendation algorithms* | User Modeling and User-Adapted Interaction (UMUAI) 2018 | 系統比較 SR（Sequential Rules）、AR、SKNN、V-SKNN 與 GRU4Rec；簡單的鄰域與規則方法常與深度模型持平或更好 | 以「上下文長度 L × 跳過 j × 位移 d」的序列規則當作主要候選來源與特徵（`candidates.py`） |
| Ludewig, Latifi & Jannach, *Empirical analysis of session-based recommendation algorithms* | UMUAI 2021 | 在 8 個資料集（含音樂 playlist／listening log）上再次確認 SKNN 系列的競爭力 | 支持「先把統計方法做到極致，再用學習排序整合」的路線 |
| Garg et al., *Sequence and Time Aware Neighborhood for Session-based Recommendations (STAN)* | SIGIR 2019 | 依「在 session 中的位置」與「時間」對鄰居 session 加權 | 以位置衰減 0.8^(j−3) 彙總 session 前段歌曲的延續訊號（特徵 `far`） |
| Jannach & Ludewig, *When Recurrent Neural Networks meet the Neighborhood for Session-Based Recommendation* | RecSys 2017 | kNN 與 GRU4Rec 互補，結合後更好 | 支持「多來源特徵 + 學習排序」的融合設計 |

## 2. 深度序列模型（調研後評估，未納入最終模型）

| 論文 | 出處 | 重點 |
|---|---|---|
| Hidasi et al., *Session-based Recommendations with Recurrent Neural Networks (GRU4Rec)* | ICLR 2016 | 首個以 RNN 做 session-based 推薦 |
| Li et al., *Neural Attentive Session-based Recommendation (NARM)* | CIKM 2017 | 注意力機制捕捉 session 主要意圖 |
| Kang & McAuley, *Self-Attentive Sequential Recommendation (SASRec)* | ICDM 2018 | Transformer 單向自注意力 |
| Wu et al., *Session-based Recommendation with Graph Neural Networks (SR-GNN)* | AAAI 2019 | 把 session 建成圖 |
| Sun et al., *BERT4Rec* | CIKM 2019 | 雙向 Transformer + Cloze 訓練 |
| Klenitskiy & Vasilev, *Turning Dross Into Gold Loss: is BERT4Rec really better than SASRec?* | RecSys 2023 | SASRec 改用 full cross-entropy 後超越 BERT4Rec |

**為何沒放進最終模型**：本資料約 77 萬首歌、71.5 萬個 session，而且 83% 的歌出現次數很少（冷啟動嚴重）。這種情況下，item embedding 很難學好。Ludewig & Jannach 的實證，以及本次第一名「GNN 效果不好，所以改用統計」的經驗，都指向同一個結論：序列規則已經把可學的訊號吃得差不多了。深度模型列為後續工作，可以當成額外的候選來源或特徵加入。

## 3. 重複消費

| 論文 | 出處 | 重點 | 本研究如何使用 |
|---|---|---|---|
| Ren et al., *RepeatNet: A Repeat Aware Neural Recommendation Machine for Session-based Recommendation* | AAAI 2019 | 把「重複 vs 探索」分開建模 | EDA 發現 17% 的目標歌已在前 20 首出現，因此 session 內所有歌都列為候選，並加入 `last_pos`、`first_pos`、`src_cnt`、`insess_cnt`、`is_last` 等特徵 |
| Li et al., *Repetition and Exploration in Sequential Recommendation* | SIGIR 2023 | 重複與探索的比例因資料而異，模型應能自適應 | 由學習排序依 session 特性（`q_uniq` 等）決定重複歌的權重 |

## 4. 兩階段「召回 + 學習排序」

| 論文 | 出處 | 重點 | 本研究如何使用 |
|---|---|---|---|
| Volkovs et al., *Two-stage Model for Automatic Playlist Continuation at Scale* | RecSys Challenge 2018（第 1 名） | Spotify 歌單續播：多來源召回後用 GBDT 重排 | 架構範本：`candidates.py`（召回）→ `train_final.py`（LightGBM 重排） |
| Ke et al., *LightGBM: A Highly Efficient Gradient Boosting Decision Tree* | NeurIPS 2017 | GBDT 實作 | 排序模型 |
| Burges, *From RankNet to LambdaRank to LambdaMART: An Overview* | MSR-TR 2010 | 學習排序 | 先試 LambdaRank；最終改用 binary（需要校準過的機率給 coverage 後處理用） |

## 5. 熱門度偏差與覆蓋率

| 論文 | 出處 | 重點 | 本研究如何使用 |
|---|---|---|---|
| Abdollahpouri, Burke & Mobasher, *Controlling Popularity Bias in Learning-to-Rank Recommendation* | RecSys 2017 | 學習排序容易偏向熱門物品，需要顯式控制長尾覆蓋 | 我們的 LightGBM 確實把不重複歌曲數壓低了（U 12.8 萬 vs 冠軍 14.9 萬），因此加入 coverage 感知重排 |
| Steck, *Calibrated Recommendations* | RecSys 2018 | 以貪婪重排在準確度與分佈校準之間取捨 | 貪婪交換：依期望 DCG 損失由小到大，把重複推薦的歌換成沒推過的候選 |

## 6. 本競賽既有解法

| 來源 | 分數 | 方法 |
|---|---|---|
| 第 1 名 Vivaldi - My_cat_can_turn_some（[writeup](https://www.kaggle.com/competitions/datagame-2023/writeups/vivaldi-my-cat-can-turn-some)、[GitHub](https://github.com/afan0918/KKCompany-Music-Challenge-Next-5-Songcraft)） | Private 0.56580（純 n-gram 0.55569，賽後 0.56691） | 位移加權 n-gram（權重 = DCG 權重）+ Jelinek-Mercer 語言模型（pyserini）補冷門歌 |
| 主辦方提示（討論區 460989、462097） | — | 樣式探勘／關聯規則；聆聽情境（歌單、電台、一起聽）決定順序 |

## 7. 調研中排除的相關資料集與基準

一開始誤以為題目是 2018 年的 **WSDM Cup – KKBox Music Recommendation Challenge**（預測 1 個月內是否重聽，評分用 AUC）。這部分的文獻（WSDM Cup 2018 overview、第 1 名 LightGBM+NN 約 0.7479、FuxiCTR/BARS 的 KKBox_x1 基準、DCNv3、DS-MLP 等）都和本題不同，因此不納入比較，細節見研究紀錄。
