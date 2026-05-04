# Lab 6 (Bonus)：把一般 LLM 訓練成 Reasoning Model（SFT 冷起動 + GRPO）

## 學習目標

本 Lab 的主軸是：**把一般的 instruct 模型，改造成具備「先思考、再回答」能力的 reasoning model**。

完成本 Lab 後，你將能夠：

1. ✅ 解釋一般 LLM 與 reasoning model（如 o1 / DeepSeek-R1）的本質差異
2. ✅ 解釋為什麼**直接做 GRPO 不夠**，需要 **SFT 冷起動 (cold start)**
3. ✅ 準備帶有 `<think>...</think>` 推理鏈的 SFT 訓練資料
4. ✅ 用 SFT 讓 `Qwen2.5-3B-Instruct` 學會以「先思考、再回答」的格式輸出
5. ✅ 在 SFT 後的模型上接續做 GRPO，強化推理正確率
6. ✅ 比較 **Base / SFT-only / SFT+GRPO** 三個階段的模型表現


**核心直覺**：

> Instruct 模型（如 Qwen2.5-3B-Instruct）原本不會主動輸出 `<think>...</think>`。
> 如果直接丟給它 GRPO，它得花大量步數「自己摸索」這個格式，
> 而且很容易陷入只追求 reward、不真正思考的捷徑。
>
> **SFT 冷起動**就像幫模型「裝上格式骨架」，
> 讓它在 GRPO 階段只需要學「怎麼想得更好」，而不是「該怎麼擺格式」。

```
┌──────────────────────────────────────────────────────────────┐
│  Stage 1：SFT Cold Start                                     │
│    用少量帶 <think>...</think> 的高品質資料 fine-tune         │
│    目標：讓模型「習慣」先思考、再回答的輸出格式              │
├──────────────────────────────────────────────────────────────┤
│  Stage 2：GRPO                                               │
│    在 SFT 模型基礎上做強化學習                                │
│    Reward = 格式分（有正確的 think tag）+ 準確分（答案對不對）│
│    目標：讓模型「想得更好」、答得更準                        │
└──────────────────────────────────────────────────────────────┘
```

## 任務設定：小學數學應用題（GSM8K）

本 Lab 統一採用 **[GSM8K](https://huggingface.co/datasets/openai/gsm8k)** 作為訓練與評測資料集，原因有三：

1. **答案唯一且為整數**，accuracy reward 極容易設計（數字相等就算對）
2. **每題都需要多步推理**（2–8 步），能明顯凸顯 `<think>` 的價值
3. **業界標準 benchmark**，未來看任何 reasoning 論文都會看到它

### GSM8K 資料集規格

| 項目 | 說明 |
|------|------|
| HuggingFace ID | `openai/gsm8k` |
| Config | `main` |
| Train split | 7,473 題 |
| Test split | 1,319 題 |
| 欄位 | `question`（題目）、`answer`（含完整推理過程，最後 `#### N` 為標準答案）|

每筆原始資料長這樣：

```
question:
  Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning
  and bakes muffins for her friends every day with four. She sells the remainder
  at the farmers' market daily for $2 per fresh duck egg. How much in dollars
  does she make every day at the farmers' market?

answer:
  Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.
  She makes 9 * 2 = $<<9*2=18>>18 every day at the farmers' market.
  #### 18
```

### 本 Lab 的資料切分（請嚴格遵守，避免資料洩漏）

| 用途 | 來源 | 大小 | 備註 |
|------|------|------|------|
| **SFT 冷起動** | `train[:200]` | 200 題 | 把 `answer` 改寫成 `<think>...</think>{最終答案}` 格式 |
| **GRPO 訓練** | `train[200:1200]` | 1,000 題 | 只用 `question` + `#### N` 的最終答案，不用推理過程 |
| **Benchmark 評測** | `test[:200]` | 200 題 | 三階段共用同一份題目，**訓練不可看** |

> 想跑得更快？把 SFT 降到 100 題、GRPO 降到 300 題、Benchmark 維持 200 題即可。
> 想對齊論文？Benchmark 用完整 `test`（1,319 題），但要跑很久。

### 期望的模型輸出格式

```
<think>
題目要算 Janet 一天賺多少錢。
她每天 16 顆蛋，扣掉早餐 3 顆、烤瑪芬 4 顆，剩下 16 - 3 - 4 = 9 顆。
每顆賣 2 美元，所以 9 * 2 = 18 美元。
</think>
18
```

> **格式約定（評測會嚴格檢查）**：
> - `<think>...</think>` 必須**完整且只出現一次**
> - `</think>` 之後就是**最終答案**，可以是純數字或一句話帶數字
> - 評測時用 regex 從 `</think>` 之後抽出最後一個整數，與 GSM8K 的 `#### N` 比對

## 模型選擇：Qwen2.5-3B-Instruct


## 檔案結構（你需要自己建立程式）

```
lab6/
├── README.md                       # 本說明文件
├── 1_prepare_sft_data.py           # 【練習】產生 SFT 冷起動資料
├── 2_sft_training.py               # 【練習】SFT 訓練
├── 3_inspect_sft_model.py          # 【練習】觀察 SFT 後模型輸出
├── 4_grpo_training.py              # 【練習】在 SFT 模型上做 GRPO
└── 5_evaluate_three_stages.py      # 【練習】比較三階段表現
```

> 本 Lab 不提供現成程式骨架，請以前面 Lab 4（GRPO）和 TRL 文件為基礎自行實作。

## 練習步驟

### Step 0：環境準備

```bash
cd lab6

# 沿用根目錄的 .venv 即可（trl / peft / transformers 已安裝）
# 如需獨立環境：
# uv venv .venv-lab6 --python 3.11
# source .venv-lab6/bin/activate
# uv pip install -r ../requirements.txt
```

確認 GPU 可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Step 1：先用 GSM8K Test 跑 Baseline

在動手訓練前，先用**評測集**（`test[:200]`）量出原始模型的分數，當作 Stage 0 baseline。

寫一段最簡單的 inference 腳本，把每題丟給 `Qwen/Qwen2.5-3B-Instruct`，記錄：

```python
from datasets import load_dataset
ds = load_dataset("openai/gsm8k", "main", split="test").select(range(200))
# for each item: item["question"] -> 模型輸出；item["answer"] 取 #### 後面的數字當 ground truth
```

並計算：

- [ ] **`format_rate`**：輸出有正確 `<think>...</think>` 結構的比例（預期：< 10%）
- [ ] **`accuracy`**：最終答案數字正確的比例（預期：30%–55%）
- [ ] **觀察**：在 system prompt 強行要求「請在 `<think>` 標籤內思考」，
  format_rate 能拉到多高？accuracy 有變化嗎？

把結果存到 `lab6/eval_results/stage0_baseline.json`，後面 Step 6 會用到。

> 這一步的目的是**建立動機**：你會發現 prompt engineering 有極限，
> 要讓模型穩定遵守特定格式並答得準，需要訓練。

### Step 2：準備 SFT 冷起動資料

在 `1_prepare_sft_data.py` 中產生**少量但高品質**的訓練資料（建議 50–200 筆）。

每筆資料的格式：

```json
{
  "messages": [
    {"role": "system", "content": "你是數學助教。請先在 <think> 標籤內逐步推理，再給出最終答案。"},
    {"role": "user", "content": "小明有 3 顆蘋果，小華有 5 顆，他們一共有幾顆？"},
    {"role": "assistant", "content": "<think>\n要算總數，把 3 和 5 相加。\n3 + 5 = 8\n</think>\n8"}
  ]
}
```

資料來源建議（擇一）：

- **方法 A**：用 GSM8K 訓練集前 100 題，**手動或用更強的模型（如 GPT-4o）改寫**成 think 格式
- **方法 B**：自製 30–50 題涵蓋加減乘除、百分比、簡單應用題
- **方法 C**：使用現成的 reasoning dataset（如 `openai/gsm8k` + 自行加上 think tag）

關鍵設計原則：

- [ ] 資料量**少而精**比多而雜更重要（DeepSeek-R1 cold start 也只用了「數千筆」）
- [ ] `<think>` 內容必須是**真實有用的推理**，不是空話
- [ ] 答案部分必須**只有最終答案**，不要重複推理過程

完成後輸出：

```
lab6/sft_data.json
```

### Step 3：執行 SFT 訓練

在 `2_sft_training.py` 中，使用 `trl.SFTTrainer` 對 `Qwen2.5-3B-Instruct` 做 LoRA SFT。

關鍵設定（建議）：

```python
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

LORA_CONFIG = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

SFT_CONFIG = {
    "output_dir": "./sft_output",
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "learning_rate": 2e-4,
    "bf16": True,
    "gradient_checkpointing": True,
    "logging_steps": 1,
    "save_strategy": "epoch",
}
```

訓練重點：

- 使用 4-bit 量化 + LoRA 節省 VRAM
- 觀察 `loss` 應在 1–2 個 epoch 內降到 < 0.5
- 把 LoRA adapter 存到 `lab6/sft_output/final/`

### Step 4：觀察 SFT 後的模型

在 `3_inspect_sft_model.py` 中，載入剛訓練好的 SFT 模型，重新問 Step 1 的同一題。

請記錄並回答：

- [ ] **Q1**：SFT 後模型是否**穩定**輸出 `<think>...</think>` 格式？
- [ ] **Q2**：think 內的內容看起來合理嗎？還是只是模仿格式、實際推理胡來？
- [ ] **Q3**：拿一道**訓練集裡沒看過**的新題目測試，模型還守得住格式嗎？答案對嗎？

> 你會發現：SFT 後模型「會說人話、會用 think」，
> 但答案**不一定對**，這正是 GRPO 要解決的事情。

### Step 5：執行 GRPO 訓練（在 SFT 模型上接續訓練）

在 `4_grpo_training.py` 中，**載入 SFT 後的 LoRA adapter** 作為 GRPO 的起點。

兩種掛載方式（擇一）：

1. **連續訓練同一份 LoRA**：`PeftModel.from_pretrained(base_model, "sft_output/final", is_trainable=True)`
2. **先 merge SFT LoRA 回 base model，再開新的 LoRA 做 GRPO**（推薦，較乾淨）

#### Reward 設計（重點！）

請至少實作兩個 reward function：

```python
def format_reward(completion: str) -> float:
    """
    檢查是否符合 <think>...</think>{answer} 格式。
    - 完整且只有一組 <think>/</think>：1.0
    - 缺少 tag：0.0
    - 多重 tag / 標籤錯位：0.3
    """
    ...

def accuracy_reward(completion: str, ground_truth: str) -> float:
    """
    從 </think> 之後抽出最終答案，與 ground truth 比對。
    - 數字完全相等：1.0
    - 答案錯誤：0.0
    """
    ...
```

組合方式（建議權重）：

```python
total_reward = 0.3 * format_reward + 0.7 * accuracy_reward
```

#### GRPO 設定參考

```python
GRPO_CONFIG = {
    "output_dir": "./grpo_output",
    "num_train_epochs": 1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 16,
    "learning_rate": 5e-6,           # GRPO 用較小 LR
    "num_generations": 4,             # 每題生成 4 個答案做組內比較
    "max_completion_length": 512,
    "temperature": 0.8,
    "beta": 0.04,                     # KL 懲罰
    "bf16": True,
    "gradient_checkpointing": True,
    "logging_steps": 1,
}
```

訓練時要觀察：

| 指標 | 期望趨勢 | 備註 |
|------|----------|------|
| `reward/mean` | ↑ 上升 | 整體變好 |
| `reward/format_reward` | 一開始就接近 1.0 | SFT 已經教會格式 |
| `reward/accuracy_reward` | ↑ 明顯上升 | GRPO 主要在優化這個 |
| `kl` | 維持小且穩定 | 不要大幅偏離 SFT 模型 |

> **關鍵觀察點**：如果你**沒做** SFT 直接做 GRPO，
> `format_reward` 一開始會非常低，整個訓練會花很多步在「學格式」上。
> SFT cold start 讓 GRPO 從 step 1 就把火力集中在「學會推理」。

### Step 6：三階段比較評估

在 `5_evaluate_three_stages.py` 中，準備一份**訓練資料中沒出現過**的 20–30 題小測驗，
用同一份題目分別測試三個版本的模型：

| 階段 | 模型 | 預期 |
|------|------|------|
| Stage 0 | 原始 `Qwen2.5-3B-Instruct` | 答案有時對，但**幾乎不會用 think tag** |
| Stage 1 | SFT 後模型 | **穩定使用 think 格式**，答案對的比例略升 |
| Stage 2 | SFT + GRPO 模型 | 格式守得住、**accuracy 明顯比 SFT 高** |

評估指標：

- `format_rate`：輸出是否包含**正確且唯一**的 `<think>...</think>` 結構
- `accuracy`：最終答案是否正確
- 平均 `<think>` 長度（觀察推理是否更詳細）

## 預期結果

完成全部步驟後，你應該能畫出類似這樣的表格：

```
                       format_rate     accuracy     think 平均長度
Base (Qwen2.5-3B)         5%            55%            -
SFT (Stage 1)            98%            62%           80 字
SFT + GRPO (Stage 2)     99%            78%          120 字
```

## 檢核點

- [ ] 能解釋「為什麼直接 GRPO 不夠」、「cold start 解決了什麼問題」
- [ ] 成功產出 `sft_data.json`，且每筆資料都有合理的 `<think>` 推理內容
- [ ] 成功跑完 SFT 訓練，存出 `sft_output/final` 的 LoRA adapter
- [ ] 觀察到 SFT 後模型**穩定輸出** `<think>...</think>` 格式
- [ ] 成功在 SFT 模型上跑完 GRPO 訓練
- [ ] 完成三階段比較表格，並能說明每階段「進步在哪裡」

## 延伸思考

1. **資料量的取捨**：DeepSeek-R1 的 cold start 只用了「幾千筆」資料，
   為什麼不直接用幾十萬筆 SFT、跳過 RL？
2. **Reward hacking**：如果你只用 `format_reward`，模型會學會什麼「壞習慣」？
3. **think 的真實性**：SFT 後模型 `<think>` 內的推理是「真實的思考」還是「演出來的」？
   要怎麼區分？GRPO 階段能改善這件事嗎？


## 常見問題

### Q：SFT loss 不下降 / 一直很高

A：檢查以下幾點：
- LoRA `r` 是否太小（試試 16 → 32）
- Learning rate 是否太低（SFT 通常用 1e-4 ~ 5e-4）
- 資料的 chat template 是否套對（用 `tokenizer.apply_chat_template` 預先檢查）

### Q：GRPO 階段 reward 反而往下掉

A：常見原因：
- `learning_rate` 太大，把 SFT 學到的格式打壞了 → 改用 1e-6 ~ 5e-6
- `beta`（KL 係數）太小，模型亂跑 → 改大到 0.04 ~ 0.1
- Reward 設計有 bug（特別是 accuracy 抽答案的 regex），先單獨測試 reward function

### Q：SFT 後模型輸出 think tag，但內容是亂七八糟的字

A：你的 SFT 資料品質不夠。請：
- 檢查是否有重複/錯誤的範例
- 增加資料的多樣性
- 或考慮用更強的模型重新生成 think 內容

---

**恭喜！** 你已經實作出一條**「SFT 冷起動 → GRPO」**的完整 pipeline，
這正是 DeepSeek-R1 等近代 reasoning 模型的核心訓練範式。
未來再看到「o1-like」、「R1-like」的論文，你就能立刻看懂他們在做什麼。
