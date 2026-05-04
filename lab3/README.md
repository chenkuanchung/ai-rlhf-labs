# Lab 3：準備 GRPO 訓練資料

## 學習目標

完成本 Lab 後，你將能夠：

1. ✅ 理解 GRPO 訓練資料的格式要求
2. ✅ 建立多種類型的訓練 prompt 資料集
3. ✅ 了解 ChatML 格式化方式

## 核心概念

### GRPO 訓練資料

**GRPO (Group Relative Policy Optimization)** 訓練只需要 prompts，不需要預先準備回答。

```
傳統 SFT 訓練：需要 (prompt, response) 配對
GRPO 訓練：只需要 prompt → 模型自己生成多個回答 → reward function 評分
```

### 資料格式

GRPO 訓練資料非常簡單，核心只需要 prompt：

```json
[
    {"prompt": "幫我查訂單 A123456789 目前狀態", "task_type": "order_query", ...},
    {"prompt": "我要查物流 TWD12345678 到哪了", "task_type": "tracking_query", ...},
    ...
]
```

### ChatML 格式

Qwen 模型使用 ChatML 格式：

```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
```

## 檔案結構

```
lab3/
├── README.md              # 本說明文件
└── prepare_dataset.py   # 【練習】準備訓練資料
```

## 練習步驟

### Step 0：環境準備

確保已安裝必要套件：

```bash
pip install trl>=0.9.0 transformers accelerate peft
```

### Step 1：準備訓練資料

執行 `prepare_dataset.py` 建立訓練用的 prompt 資料集：

```bash
cd lab3
uv run prepare_dataset.py
```

執行後會產生 `training_prompts.jsonl`，包含多種類型的訓練 prompt。

## 資料類型說明

| 類型 | 說明 | 期望行為 |
|------|------|----------|
| `order_query` | 查詢訂單狀態 | 呼叫 `get_order_status` |
| `tracking_query` | 查詢物流狀態 | 呼叫 `track_shipment` |
| `refund_incomplete` | 退款（資訊不全） | 追問，不呼叫工具 |
| `refund_complete` | 退款（資訊完整） | 呼叫 `create_refund_request` |

## 預期結果

執行完成後，你應該看到：

1. ✅ 產生 `training_prompts.jsonl` 檔案
2. ✅ 共 75 筆 prompt（30 訂單 + 20 物流 + 10 追問退款 + 15 完整退款）
3. ✅ 各類型統計正確

## 檢核點

- [ ] 成功執行 `prepare_dataset.py`
- [ ] 產生 `training_prompts.json`
- [ ] 理解 GRPO 只需要 prompt 不需要 response
- [ ] 理解 ChatML 格式

## 延伸思考
1. 如何產生更多訓練資料？

---

**下一步**：Lab 4 將使用這份資料進行 GRPO 訓練。
