# AI 新秀計畫: 大語言模型推理強化實作 (Course 8)

本專案為大語言模型（LLM）後訓練（Post-training）的實作紀錄，重點探討如何將一般的 Instruct 模型，透過 **Supervised Fine-Tuning (SFT)** 與 **Group Relative Policy Optimization (GRPO)** 演算法，改造成具備「先思考、再回答」能力的 Reasoning Model（類 DeepSeek-R1 架構）。

## 🎓 致謝與版權宣告

本專案的基礎架構、教材與核心指導均來自以下開源資源與講師，在此表達誠摯的感謝：

* **教材提供與指導**：群聯電子 陳界安老師
* **原始碼來源 / Upstream Repository**：[Alkalid/2026_ai_rookie_course8](https://github.com/Alkalid/2026_ai_rookie_course8)
* **實作者**：陳冠中 (於原框架上進行實作、除錯與延伸整合)

---

## 💡 專案核心目標

現代 LLM 在處理複雜邏輯題時，若只依賴單次生成往往容易出錯。本專案透過實作，驗證了以下核心概念：
1. **SFT 冷起動 (Cold Start)**：為模型裝上 `<think>...</think>` 的格式骨架，避免強化學習初期迷失方向。
2. **GRPO 強化學習**：基於 SFT 建立的基礎，設計 `format_reward` 與 `accuracy_reward`，讓模型在無監督情況下自我進化，提升數學推理的準確率。
3. **高效能推論**：整合 `vLLM` 引擎與 Docker 容器化技術，並透過 LoRA 進行輕量化掛載與推論。

---

## 🛠️ 技術棧 (Tech Stack)

* **基礎模型**：`Qwen/Qwen3.5-2B` / `Qwen/Qwen2.5-3B-Instruct`
* **訓練框架**：Hugging Face `transformers`, `trl` (SFTTrainer, GRPOTrainer), `peft` (LoRA)
* **優化技術**：QLoRA (4-bit nf4 量化), Gradient Checkpointing, Flash Attention
* **推論引擎**：`vLLM` (搭配 Docker 部署)
* **套件管理**：`uv`

---

## 📂 實作目錄總覽

專案分為多個循序漸進的 Lab，帶領我們從基準測試走到最終的強化學習：

* **`lab1/` - 基準測試 (Baseline Evaluation)**
  * 使用原始模型對 GSM8K 數學題目進行測試，建立 Format Rate 與 Accuracy 的初始基準分數。
* **`lab3/` - vLLM 推論伺服器架設**
  * 撰寫 `docker-compose.yml`，利用 vLLM 建立高效能的 OpenAI 相容 API 伺服器，並學習資源分配（VRAM 控管）。
* **`lab4/` - GRPO 強化學習初探**
  * 撰寫 Reward Functions，實作 TRL 的 `GRPOTrainer`，讓模型從錯誤中學習。
* **`lab5/` - 訓練後權重驗證**
  * 將 Lab 4 訓練出的 LoRA 權重透過 vLLM 掛載（`--enable-lora`），並設計推論腳本自動過濾 `<think>` 標籤，比較訓練前後的效能差異。
* **`lab6/` - 打造終極推理模型 (SFT + GRPO)**
  * **Step 1:** 資料清理與格式化（產出 `<think>` 格式的教學資料）。
  * **Step 2:** SFT 冷起動（教導模型正確的思考格式）。
  * **Step 3:** 載入 SFT 權重接續進行 GRPO 訓練（專注提升推理準確率）。
  * **Step 4:** 自動化三階段評估（Base vs. SFT vs. GRPO），產出量化報表。

---

## ⚙️ 環境建置與使用說明

本專案使用 `uv` 進行快速的 Python 套件管理。

### 1. 安裝環境
```bash
# 建立並啟動虛擬環境
uv venv .venv --python 3.11
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安裝依賴套件
uv pip install -r requirements.txt
```

### 2. 啟動 vLLM 伺服器 (Lab 3 / Lab 5)
```bash
# 在含有 docker-compose.yml 的目錄下執行
docker compose up -d

# 查看日誌確認服務啟動與 LoRA 載入狀態
docker compose logs -f
```

### 3. 執行訓練與評估 (以 Lab 6 為例)
```bash
cd lab6

# 1. 準備 SFT 資料
uv run 1_prepare_sft_data.py

# 2. 執行 SFT 訓練 (需 GPU)
uv run 2_sft_training.py

# 3. 執行 GRPO 訓練 (需 GPU)
uv run 4_grpo_training.py

# 4. 三階段終極評估
uv run 5_evaluate_three_stages.py
```

## 📊 預期成果 (以 GSM8K 為例)

經過完整的 Pipeline 訓練後，模型在數學推理任務上的表現將有顯著提升：

| 模型階段 | Format Rate | Accuracy | 特徵描述 |
| :--- | :--- | :--- | :--- |
| **Base 模型** | 極低 | 普通 | 幾乎不使用思考標籤，容易算錯 |
| **SFT 模型** | ~98% | 略微提升 | 能穩定輸出 `<think>` 格式，具備初步推論骨架 |
| **GRPO 模型** | ~99% | **大幅提升** | 推論過程更詳盡（平均思考字數增加），答案正確率最高 |








