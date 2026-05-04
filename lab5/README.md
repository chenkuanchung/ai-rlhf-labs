# Lab 5：評估訓練成果

## 學習目標

完成本 Lab 後，你將能夠：

1. ✅ 載入並使用訓練後的模型（LoRA adapter）產生推論結果
2. ✅ 使用與 Lab 1 相同的評估腳本，量化訓練前後的差異
3. ✅ 理解如何把前面所有 Lab 串成一套完整的評估流程

## 核心概念

### 為什麼要評估？

訓練時看到 `reward/mean` 上升不代表模型真的變好了，我們需要用**獨立的測試案例**來驗證：

```
訓練資料 → 模型學習的依據（不應拿來評估）
測試資料 → 獨立的案例，用來驗證模型是否真正學會
```

為了讓「訓練前」和「訓練後」可以公平比較，本 Lab 直接沿用 **Lab 1 的測試集與評估腳本**。

### 評估指標（與 Lab 1 完全一致）

| 指標 | 說明 |
|------|------|
| `valid_rate` | 格式合法率（tool_call 能通過 JSON Schema 驗證的比例） |
| `tool_acc` | 工具選擇準確率 |
| `args_exact` | 參數完全相符率（僅計算有 `arguments` 的案例） |

> Lab 1 已經跑出了「訓練前」的 baseline 數值，本 Lab 的任務就是用「訓練後」的模型重跑一次，並比較兩者的差異。

## 前置作業

1. 已完成 Lab 1，並產生 `lab1/baseline_outputs.json` 與 `lab1/eval_report.json`（訓練前的結果）
2. 已完成 Lab 4 的 GRPO 訓練，確認訓練後的模型已儲存：

```bash
# 確認 lab4/grpo_output/final 目錄存在
ls lab4/grpo_output/final
```

## 檔案結構

```
lab5/
└── README.md    # 本說明文件（本 Lab 沒有提供現成程式，請自己動手）
```

## 練習步驟

本 Lab 不提供現成的評估腳本，請你自己動手完成下列流程。

### Step 1：用訓練後的模型產生推論結果

目標是產生一份**格式與 `lab1/baseline_outputs.json` 相同**的 JSON 檔，裡面每一筆長這樣：

```json
{
  "id": "c1",
  "messages": [{"role": "user", "content": "幫我查訂單 A123456789 目前狀態"}],
  "predict": "模型的原始輸出字串",
  "expect": {"tool": "get_order_status", "arguments": {"order_id": "A123456789"}}
}
```

你可以沿用 `lab1/1_baseline_inference.py` 的架構，關鍵差別只在於 **inference 改用訓練後的模型**。常見做法（擇一）：

- **方法 A（推薦）**：用 vLLM 把 `lab4/grpo_output/final` 的 LoRA adapter 載入後啟動服務，再讓 `call_llm` 呼叫該服務
- **方法 B**：自行用 `transformers` + `peft` 載入基礎模型 + LoRA adapter，在本機跑 inference
- **方法 C**：把 LoRA adapter 先 merge 回基礎模型，再用 vLLM 部署

請把輸出的 JSON 放到一個你自己決定的位置，例如：

```
lab5/trained_outputs.json
```

### Step 2：用 Lab 1 的評估腳本評估結果

`lab1/2_evaluate.py` 預設會讀 `lab1/baseline_outputs.json` 並輸出 `eval_report.json`。你有兩種做法：

1. 把 Step 1 產出的 JSON 檔放到 `lab1/baseline_outputs.json` 的位置（記得**先備份**原本訓練前的結果），再直接執行 `uv run 2_evaluate.py`
2. 或是稍微調整 `2_evaluate.py` 的輸入/輸出路徑，讓它讀你剛產生的 `lab5/trained_outputs.json`，輸出 `lab5/eval_report_trained.json`

### Step 3：比較訓練前後的指標

把 Lab 1 的結果（訓練前）與 Lab 5 的結果（訓練後）放在一起比較：

```
指標                 訓練前       訓練後         變化
-------------------------------------------------------
格式合法率     (valid_rate)    62.5%     ...%      ↑ ...
工具選擇準確率   (tool_acc)    78.1%     ...%      ↑ ...
參數完全相符率 (args_exact)    89.5%     ...%      ↑ ...
```

> 小提示：訓練前的 baseline 數字可以直接參考 `lab1/README.md` 的「預期產出」，或打開 `lab1/eval_report.json` 看 `summary` 區塊。

## 預期結果

評估完成後，你應該看到：

1. ✅ 訓練後模型的 `valid_rate` 高於訓練前
2. ✅ `tool_acc` 與 `args_exact` 至少不退步，理想情況下有所提升
3. ✅ 能夠說明「哪些案例從錯變對、哪些從對變錯、為什麼」

## 檢核點

- [ ] 成功用訓練後的模型產生 `trained_outputs.json`（格式與 `lab1/baseline_outputs.json` 一致）
- [ ] 成功用 `lab1/2_evaluate.py` 對訓練後結果算出 `valid_rate` / `tool_acc` / `args_exact`
- [ ] 能列出訓練前後三項指標的數值與差異
- [ ] 能指出至少一個「訓練後明顯改善」的案例

## 延伸思考

1. 如果訓練後模型反而變差，可能的原因是什麼？（提示：想想 Lab 4 的 `reward/mean` 與 `kl`）
2. 32 個測試案例夠嗎？理想的測試案例數量是多少？該怎麼擴充？
3. 除了目前的三個指標，還有哪些指標值得追蹤？（例如：回應長度、推論時間、幻覺率⋯⋯）
4. 訓練前後某些案例「從對變錯」，這代表什麼？要怎麼處理？

---

**恭喜完成所有 Lab！** 你已經完整走過了 RLHF/GRPO 的流程：

- Lab 1：認識 LLM 與工具呼叫，建立 baseline
- Lab 2：設計 Reward Function
- Lab 3：準備訓練資料
- Lab 4：執行 GRPO 訓練
- Lab 5：用 Lab 1 同一套指標評估訓練成果
