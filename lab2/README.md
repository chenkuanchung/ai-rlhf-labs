# Lab 2：資料集與 Reward Function

## 學習目標

完成本 Lab 後，你將能夠：

1. ✅ 理解 GRPO 訓練需要的資料格式
2. ✅ 把「偏好」概念轉換為可計算的 reward function
3. ✅ 實作格式檢查與工具正確性的 reward function
4. ✅ 理解多個 reward 如何疊加組合

## 核心概念

### 什麼是 Reward Function？

在 RLHF 中，我們不是告訴模型「正確答案是什麼」，而是教模型「什麼樣的回答比較好」。

```
傳統監督學習：這個輸入的正確輸出是 X
RLHF 思維：   這個輸出比那個輸出「更好」（更符合我們的偏好）
```

**Reward Function** 就是把「好不好」量化成數字的函式：
- 回傳值越高 → 越符合偏好
- 回傳值越低 → 越不符合偏好

### GRPO 資料格式

GRPO（Group Relative Policy Optimization）訓練需要：

```
對於每個 prompt：
    1. 讓模型生成多個 completions（例如 4 個）
    2. 用 reward function 對每個 completion 打分
    3. 模型學習：偏好高分的 completion，避免低分的 completion
```

資料格式範例：

```json
{
  "prompt": "幫我查訂單 A123456789 目前狀態",
  "completions": [
    {"text": "{\"type\":\"tool_call\",\"name\":\"get_order_status\",\"arguments\":{\"order_id\":\"A123456789\"}}", "reward": 1.0},
    {"text": "好的，我來幫你查詢訂單狀態...", "reward": 0.2},
    {"text": "{\"name\":\"get_order\",\"args\":{}}", "reward": 0.3},
    {"text": "{invalid json", "reward": 0.0}
  ]
}
```

## 檔案結構

```
lab2/
├── README.md                    # 本說明文件
├── 1_reward_functions.py        # 【練習】Reward 函式實作
├── 2_test_rewards.py            # 【練習】測試 Reward 函式
└── sample_completions.json     # 範例資料（用於測試）
```

## 練習步驟

### Step 1：理解 Reward 設計原則

好的 Reward Function 應該：

1. **可計算**：能自動判斷，不需人工介入
2. **可微分**：分數有層次（不只是 0/1）
3. **對齊目標**：真正反映我們想要的行為

### Step 2：實作格式 Reward

完成 `1_reward_functions.py` 中的 `format_reward()` 函式：

```python
def format_reward(response: str) -> float:
    """
    評估回應的 JSON 格式正確性
    
    評分標準：
    - 0.0：無法解析為 JSON
    - 0.3：是 JSON 但缺少必填欄位
    - 0.7：有必填欄位但格式不完全正確
    - 1.0：完全符合 tool_call 格式
    """
```

### Step 3：實作工具正確性 Reward

完成 `tool_correctness_reward()` 函式：

```python
def tool_correctness_reward(response: str, expected_tool: str, expected_args: dict) -> float:
    """
    評估工具選擇和參數的正確性
    
    評分標準：
    - 0.0：JSON 解析失敗
    - 0.3：JSON 正確但工具名稱錯誤
    - 0.6：工具名稱正確但參數錯誤
    - 1.0：工具和參數都正確
    """
```

### Step 4：組合多個 Reward

了解如何將多個 reward 疊加：

```python
def combined_reward(response: str, expected_tool: str, expected_args: dict) -> float:
    """
    組合多個 reward function
    
    total = w1 * format_reward + w2 * tool_reward
    """
```

### Step 5：執行測試

```bash
cd lab2
uv run 2_test_rewards.py
```

## Reward 設計指南

### 格式 Reward 的層次設計

| 分數 | 條件 | 說明 |
|------|------|------|
| 0.0 | 無法解析 | 完全不是 JSON |
| 0.3 | 是 JSON 物件 | 至少是合法的 JSON |
| 0.5 | 有 `name` 欄位 | 開始接近正確格式 |
| 0.7 | 有 `name` + `arguments` | 結構大致正確 |
| 1.0 | 完全符合 schema | 完美格式 |

### 為什麼要有層次？

如果只用 0/1 評分：
- 模型很難學習「接近正確」的價值
- 訓練初期幾乎全是 0 分，梯度訊號弱

有層次的評分：
- 「部分正確」也能得到部分分數
- 模型能逐步改進，而非只能碰運氣

## 預期產出

執行完成後，你應該能夠：

1. ✅ 理解 reward function 的設計思路
2. ✅ 實作格式檢查 reward
3. ✅ 實作工具正確性 reward
4. ✅ 理解如何組合多個 reward

## 檢核點

- [ ] 能解釋 reward function 的作用
- [ ] format_reward 能區分不同格式錯誤
- [ ] tool_correctness_reward 能正確評分
- [ ] 理解 reward 疊加的權重設計

## 延伸思考

1. 如果要加入「簡潔性」reward（回答越短分數越高），該怎麼設計？
2. 如何處理「追問」場景的 reward？（模型應該追問而非亂猜）
3. Reward hacking 是什麼？如何避免模型「鑽漏洞」取得高分？

---


