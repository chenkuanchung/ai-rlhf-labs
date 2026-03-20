# Lab 6：使用開源專案跑 GRPO 訓練

## 學習目標

完成本 Lab 後，你將能夠：

1. ✅ 讀懂真實開源 GRPO 專案的結構與運作方式
2. ✅ 理解 YAML config 驅動的訓練流程設計
3. ✅ 使用 Open R1 執行 GRPO 訓練
4. ✅ 理解 reward function registry 的設計模式

## 背景

在 Lab 2–5 中，我們從零開始手刻了 GRPO 的訓練流程——那像是在「玩玩具」，幫助你理解核心概念。

但在真實工作中，你不會從頭寫訓練迴圈，而是會使用**成熟的開源專案**。本 Lab 將帶你走過 Hugging Face 官方的 [**Open R1**](https://github.com/huggingface/open-r1) 專案，這是一個完整復現 DeepSeek-R1 訓練流程的開源實作。

```
Lab 3-5（玩具版）              →    Lab 6（真實專案）
─────────────────────         ─────────────────────
手刻 training loop             使用 GRPOTrainer + accelerate
手動定義 reward function       Reward function registry
直接寫死參數                   YAML config 管理
單 GPU                        支援多 GPU / 多節點
```

## Open R1 專案簡介

Open R1 由 Hugging Face 開發，目標是完整復現 DeepSeek-R1 的訓練流程。

### 核心架構

```
open-r1/
├── src/open_r1/
│   ├── grpo.py          # GRPO 訓練主程式（進入點）
│   ├── sft.py           # SFT 訓練主程式
│   ├── rewards.py       # Reward function 定義（重點！）
│   ├── configs.py       # 訓練參數定義
│   └── utils/           # 工具函式
├── recipes/             # 訓練設定檔（YAML）
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/grpo/
│   │   └── config_demo.yaml
│   ├── Qwen2.5-1.5B-Instruct/grpo/
│   │   └── config_demo.yaml
│   └── accelerate_configs/
│       ├── zero2.yaml
│       └── zero3.yaml
└── Makefile             # 快捷指令
```

## 練習步驟

### Step 0：環境準備

```bash
cd lab6/open-r1

# 建立虛擬環境
uv venv openr1 --python 3.11
source openr1/bin/activate  # Linux/Mac
# Windows: openr1\Scripts\activate

# 安裝依賴
uv pip install --upgrade pip
uv pip install vllm==0.8.5.post1
uv pip install setuptools && uv pip install flash-attn --no-build-isolation
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]"
```

### Step 1：閱讀原始碼（原始碼導讀）

> **這是本 Lab 最重要的步驟。** 不要急著跑訓練，先讀懂程式碼。

#### 1.1 訓練主程式：`src/open_r1/grpo.py`

這個檔案只有約 180 行，卻完成了整個 GRPO 訓練流程。請仔細閱讀並回答以下問題：

- [ ] **Q1**：`grpo.py` 的 `main()` 做了哪些步驟？（列出 5 個以上）
- [ ] **Q2**：它怎麼載入 reward function？（提示：看 `get_reward_funcs`）
- [ ] **Q3**：dataset 的 prompt 是怎麼格式化的？（提示：看 `make_conversation`）

```
💡 對比 Lab 4 的手刻版本：
   Lab 4：手動寫 load model → load data → define reward → train → save
   Open R1：同樣的流程，但用 config + registry 模式組織
```

#### 1.2 Reward Functions：`src/open_r1/rewards.py`

這是整個專案最值得學習的檔案之一。它展示了如何用 **registry 模式** 管理多個 reward function。

請閱讀並回答：

- [ ] **Q4**：`accuracy_reward` 怎麼判斷答案是否正確？（提示：用了 `math_verify` 套件）
- [ ] **Q5**：`format_reward` 檢查什麼格式？期望的輸出結構是什麼？
- [ ] **Q6**：`tag_count_reward` 和 `format_reward` 有什麼差異？為什麼需要兩個？
- [ ] **Q7**：`get_reward_funcs()` 怎麼把 YAML 中的字串（如 `"accuracy"`）對應到真正的函式？

```python
# rewards.py 中的 registry 模式（關鍵設計）
REWARD_FUNCS_REGISTRY = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "tag_count": tag_count_reward,
    "reasoning_steps": reasoning_steps_reward,
    ...
}
# 從 config 讀取需要哪些 reward → 查表取得函式
reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
```

#### 1.3 訓練設定檔：YAML Config

請閱讀 `recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml`，回答：

- [ ] **Q8**：這個 config 用了哪些 reward function？各自的 weight 是多少？
- [ ] **Q9**：`num_generations: 16` 代表什麼？和 Lab 4 的 `num_generations: 4` 相比，差異是什麼？
- [ ] **Q10**：`max_prompt_length` 和 `max_completion_length` 分別控制什麼？

### Step 2：理解 Config 驅動的設計

Open R1 用 YAML config 管理所有訓練參數，這是業界常見的設計模式：

```yaml
# recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml 重點節錄

# 模型設定
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct

# 資料設定
dataset_name: open-r1/OpenR1-Math-220k
dataset_prompt_column: problem
system_prompt: "You are a helpful AI Assistant..."

# Reward 設定（可以組合多個！）
reward_funcs:
- accuracy      # 答案正確性
- format        # 格式是否符合 <think>...</think><answer>...</answer>
- tag_count     # 標籤數量是否正確
reward_weights:
- 1.0
- 1.0
- 1.0

# 訓練超參數
num_generations: 16
learning_rate: 2.0e-05
per_device_train_batch_size: 16
gradient_accumulation_steps: 4
```

**對比 Lab 4 的硬編碼方式：**

```python
# Lab 4：參數直接寫在程式碼裡
GRPO_CONFIG = {
    "num_generations": 4,
    "learning_rate": 5e-5,
    ...
}
```

- [ ] **Q11**：Config 驅動有什麼好處？（提示：想想實驗管理、版本控制、團隊協作）

### Step 3：執行 GRPO 訓練

> ⚠️ 需要 GPU 環境（建議 VRAM >= 24GB）

#### 方式一：使用 vLLM colocate 模式（單節點）

```bash
cd lab6/open-r1

ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml \
    --vllm_mode colocate
```

#### 方式二：手動啟動 vLLM server（適合 debug）

終端機 1 - 啟動 vLLM server：

```bash
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct
```

終端機 2 - 啟動訓練：

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 \
    src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml
```

### Step 4：觀察訓練日誌

訓練過程中觀察以下指標：

| 指標 | 說明 | 期望趨勢 |
|------|------|----------|
| `reward/accuracy` | 答案正確率 | ↑ 上升 |
| `reward/format` | 格式正確率 | ↑ 上升 |
| `reward/tag_count` | 標籤正確率 | ↑ 上升 |
| `kl` | KL 散度 | 維持穩定 |
| `loss` | 訓練損失 | ↓ 下降 |

如果有設定 Weights & Biases，可以在 wandb dashboard 上看到即時圖表。

## 重點對比：Lab 4 vs Open R1

| 面向 | Lab 4（手刻版） | Open R1（開源專案） |
|------|-----------------|---------------------|
| **程式碼結構** | 單一 Python 檔案 | 模組化（grpo.py, rewards.py, configs.py） |
| **參數管理** | 寫死在程式碼中 | YAML config + CLI 覆蓋 |
| **Reward Function** | 直接定義 | Registry 模式，config 指定 |
| **模型載入** | 手動 + LoRA | 自動處理（含量化、LoRA、PEFT） |
| **分散式訓練** | 不支援 | DeepSpeed ZeRO-2/3 + 多節點 |
| **生成加速** | 無 | vLLM backend |
| **實驗追蹤** | 無 | Weights & Biases 整合 |

## 檢核點

### 原始碼閱讀（必做）

- [ ] 讀完 `grpo.py`，能說出訓練流程的 5 個步驟
- [ ] 讀完 `rewards.py`，能解釋 `accuracy_reward` 和 `format_reward` 的差異
- [ ] 讀完 YAML config，理解每個參數的意義
- [ ] 回答 Q1–Q11

### 實際執行（選做，需 GPU）

- [ ] 成功啟動 GRPO 訓練
- [ ] 在日誌中觀察到 reward 上升
- [ ] 嘗試修改 config 參數（如 `learning_rate`、`num_generations`）重新訓練

## 延伸思考

1. **Reward 組合**：Open R1 支援同時使用多個 reward function 並給予不同權重。為什麼要這麼做？如果只用 `accuracy_reward`，可能會有什麼問題？
2. **vLLM 加速**：為什麼 Open R1 要用 vLLM 來做生成？和 Lab 4 直接用 `model.generate()` 相比，效能差多少？
3. **DeepSpeed ZeRO**：`accelerate_configs/` 裡有 `zero2.yaml` 和 `zero3.yaml`，它們分別適用什麼場景？
4. **想想看**：如果要把 Lab 2 寫的 tool-calling reward function 加進 Open R1 的 registry，你需要修改哪些檔案？

## 常見問題

### Q：跑不起來 / OOM

A：嘗試調小以下參數：
- `num_generations`：16 → 4
- `per_device_train_batch_size`：16 → 1
- `max_completion_length`：1024 → 256
- 切換到 `zero3.yaml`（記憶體更省）

### Q：沒有 GPU 怎麼辦？

A：本 Lab 的**原始碼閱讀**部分不需要 GPU。請完成 Step 1 和 Step 2 的所有問題，這才是最重要的學習。

### Q：vLLM 安裝失敗

A：vLLM 需要 CUDA 12.4+，請確認你的環境：
```bash
nvcc --version
```

---

**恭喜！** 你已經從「手刻玩具」走到「使用真實開源專案」，具備了在實際工作中跑 GRPO 訓練的能力。
