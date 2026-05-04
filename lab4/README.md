# Lab 4：GRPO 訓練跑通

## 學習目標

完成本 Lab 後，你將能夠：

1. ✅ 理解 GRPO 的核心概念（無需數學細節）
2. ✅ 使用 Hugging Face TRL 的 GRPOTrainer 進行訓練
3. ✅ 觀察 reward 指標上升，驗證訓練有效

## 核心概念（最少必要知識）

### GRPO 是什麼？

**GRPO (Group Relative Policy Optimization)** 是一種讓模型學習「偏好」的訓練方法。

```
傳統訓練：告訴模型「正確答案是 X」
GRPO 訓練：告訴模型「A 比 B 好，B 比 C 好」→ 模型學會偏好 A
```

### GRPO 運作流程

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1：對每個 prompt 生成多個回答                              │
│                                                                 │
│    Prompt: "查訂單 A123"                                        │
│        ↓                                                        │
│    生成 4 個不同的回答                                          │
│        Response 1: {"type":"tool_call",...}  ← 完美格式         │
│        Response 2: {"name":"get_order",...}  ← 缺少 type        │
│        Response 3: "好的，我來查..."          ← 沒用工具         │
│        Response 4: {invalid json}            ← 格式錯誤         │
├─────────────────────────────────────────────────────────────────┤
│  Step 2：用 Reward Function 評分                                │
│                                                                 │
│        Response 1: reward = 1.0  ★★★★★                         │
│        Response 2: reward = 0.7  ★★★☆☆                         │
│        Response 3: reward = 0.2  ★☆☆☆☆                         │
│        Response 4: reward = 0.0  ☆☆☆☆☆                         │
├─────────────────────────────────────────────────────────────────┤
│  Step 3：更新模型                                               │
│                                                                 │
│    讓「高分回答」出現機率 ↑                                      │
│    讓「低分回答」出現機率 ↓                                      │
│                                                                 │
│    → 模型逐漸學會輸出正確格式！                                  │
└─────────────────────────────────────────────────────────────────┘
```

### KL 散度：別讓模型跑太遠

訓練時，我們希望模型學會新行為，但又不要完全忘記原本的能力。

**KL 散度 (KL Divergence)** 衡量「新模型」和「原始模型」的差異：
- KL 太大 → 模型偏離太遠，可能學壞
- KL 太小 → 模型幾乎沒變，學習效果差

```
KL 懲罰就像「橡皮筋」：
- 允許模型往目標方向移動
- 但拉太遠就會被彈回來
- 確保訓練穩定，不會崩壞
```

## 前置作業

請先完成 Lab 3，產生訓練資料：

```bash
cd lab3
python 1_prepare_dataset.py
```

確認 `lab3/training_prompts.jsonl` 存在。

## 檔案結構

```
lab4/
├── README.md                    # 本說明文件
├── 1_grpo_training.py           # 【練習】GRPO 訓練主程式
└── 1_grpo_training_simple.py    # 【參考】簡化版訓練（更容易理解）
```

## 練習步驟

### Step 1：執行 GRPO 訓練

執行 `1_grpo_training.py` 開始訓練：

```bash
cd lab4
uv run 1_grpo_training.py
```

訓練過程中觀察：
- `reward/mean`：平均 reward，應該逐漸上升
- `kl`：KL 散度，應該維持在合理範圍
- `loss`：損失值

> **備註**：如果完整版太複雜或遇到問題，可以先看簡化版 `1_grpo_training_simple.py`

## 關鍵參數說明

| 參數 | 說明 | 建議值 |
|------|------|--------|
| `num_generations` | 每個 prompt 生成幾個回答 | 4 |
| `max_new_tokens` | 最大生成 token 數 | 256 |
| `learning_rate` | 學習率 | 1e-5 ~ 5e-5 |
| `kl_coef` | KL 懲罰係數 | 0.01 ~ 0.1 |
| `num_train_epochs` | 訓練輪數 | 1 ~ 3 |

## 預期結果

訓練完成後，你應該看到：

1. ✅ `reward/mean` 從 ~0.8 上升到 ~0.9 以上
2. ✅ 模型儲存在 `grpo_output/final`
3. ✅ 訓練過程中沒有 OOM 錯誤

## 檢核點

- [ ] 成功執行訓練，無 OOM 錯誤
- [ ] 觀察到 reward 指標上升
- [ ] 理解 GRPO 的運作原理
- [ ] 理解 KL 散度的作用
- [ ] （Bonus）嘗試調整訓練參數，例如增加num_generations
- [ ] （Bonus）嘗試優化獎勵函式，或新增其他的獎勵函式

## 常見問題

### Q1：OOM (Out of Memory)

A：嘗試以下方法：
- 減少 `num_generations`（例如 4 → 2）
- 減少 `per_device_train_batch_size`
- 啟用 gradient checkpointing
- 使用 LoRA 減少記憶體需求

### Q2：Reward 沒有上升

A：可能原因：
- 學習率太低，嘗試調高
- 訓練步數太少，增加 epochs
- Reward function 設計問題，檢查 Lab 2

### Q3：KL 太高

A：模型偏離太遠，嘗試：
- 降低學習率
- 減少訓練步數

## 延伸思考

1. 為什麼 GRPO 需要同時生成多個回答？（對比學習）
2. 如果只有兩個回答（好/壞），和多個回答有什麼差異？
3. KL 係數設太大或太小會發生什麼事？

---

**下一步**：Lab 5 將評估訓練後的模型表現，比較訓練前後的差異。
