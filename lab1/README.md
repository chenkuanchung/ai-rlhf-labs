# Lab 1：環境設定與 Baseline 推論

## 學習目標

完成本 Lab 後，你將能夠：

1. ✅ 成功載入並執行 Qwen3.5-4B 模型推論
2. ✅ 理解「對齊目標」是可量測的
3. ✅ 產生 baseline 輸出，作為訓練前的對照基準

## 任務說明

### 背景

在進行 RLHF（Reinforcement Learning from Human Feedback）訓練之前，我們需要先建立一個 **Baseline**，也就是記錄模型「訓練前」的表現。這樣在訓練後，我們才能比較模型是否有進步。

本次課程的對齊目標是：**格式對齊** —— 讓模型正確輸出 Tool Call JSON 格式。

### 任務

使用 `eval_cases.json` 中的測試案例，對模型進行推論，並將結果儲存為 `baseline_outputs.json`。

## 檔案結構

```
lab1/
├── README.md                  # 本說明文件
├── eval_cases.json            # 評估測試案例（已提供，共 32 筆）
├── 1_baseline_inference.py    # 【練習】Baseline 推論程式
├── 2_evaluate.py              # 【練習】評估腳本
├── baseline_outputs.json      # 【產出】推論結果（執行後產生）
└── eval_report.json           # 【產出】評估報告（執行後產生）
```

## 練習步驟

### Step 1：確認環境

確保 vLLM 服務已啟動：

```bash
# 檢查服務是否運行
curl http://localhost:8299/v1/models
```

### Step 2：理解測試案例格式

查看 `eval_cases.json` 的內容：

```json
{
  "id": "c1",
  "messages": [{"role": "user", "content": "幫我查訂單 A123456789 目前狀態"}],
  "expect": {
    "tool": "get_order_status",
    "arguments": {"order_id": "A123456789"}
  }
}
```

- `id`：案例編號
- `messages`：對話內容（輸入給模型的 prompt）
- `expect`：期望的輸出（用於評估）

### Step 3：完成 1_baseline_inference.py

請完成 `1_baseline_inference.py` 中標記 `TODO` 的部分：

1. 載入測試案例
2. 對每個案例進行推論
3. 儲存推論結果

### Step 4：執行推論

```bash
cd lab1
uv run 1_baseline_inference.py
```

成功執行後，會產生 `baseline_outputs.json`。

### Step 5：執行評估

```bash
uv run 2_evaluate.py
```

查看模型的 baseline 表現指標。

## 評估指標說明

| 指標 | 說明 |
|------|------|
| `valid_rate` | 格式合法率（tool_call 能通過 JSON Schema 驗證的比例）|
| `tool_acc` | 工具選擇準確率 |
| `args_exact` | 參數完全相符率（僅計算有 arguments 的案例）|

## 預期產出

執行完成後，你應該有：

1. ✅ `baseline_outputs.json`：包含 32 筆推論結果
2. ✅ `eval_report.json`：詳細評估報告（含每筆案例結果）
3. ✅ 評估結果接近以下 baseline 數值：

```
格式合法率     (valid_rate)：62.5%
工具選擇準確率   (tool_acc)：78.1%
參數完全相符率 (args_exact)：89.5%
```

## 檢核點

- [ ] 能成功執行推論，無 OOM 錯誤
- [ ] 產生 `baseline_outputs.json`（32 筆）與 `eval_report.json`
- [ ] 能理解各項評估指標的意義

## 常見問題

### Q1：連線失敗 (Connection refused)

A：確認 vLLM 服務是否啟動，檢查 port 是否正確（預設 8299）。

### Q2：OOM (Out of Memory)

A：調低 `max_tokens` 或使用更小的 batch size。

### Q3：JSON 解析失敗

A：這是正常的！Baseline 模型可能還不太會輸出正確的 JSON 格式，這正是我們後續要對齊的目標。

## 延伸思考

1. 觀察 baseline 的輸出，模型最常犯的錯誤是什麼？
2. 如果你要設計一個 reward function，會怎麼評估「格式正確性」？
3. 除了格式對齊，還有哪些可能的對齊目標？

---

**下一步**：Lab 2 將介紹如何設計 Reward Function，為 RLHF 訓練做準備。
