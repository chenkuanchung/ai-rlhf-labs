"""
==============================================================================
Lab 4：GRPO 訓練主程式
==============================================================================

📌 本檔案功能：
    使用 Hugging Face TRL 的 GRPOTrainer 進行 GRPO 訓練。
    
    訓練流程：
    1. 載入預訓練模型（Qwen3.5-4B）
    2. 設定 LoRA 進行高效微調
    3. 定義 Reward Function
    4. 執行 GRPO 訓練
    5. 儲存訓練後的模型

📖 GRPO 核心概念：
    - 對每個 prompt 生成多個回答
    - 用 reward function 對回答評分
    - 更新模型，讓高分回答更常出現

🔧 使用方式：
    cd lab4
    python 1_grpo_training.py

"""

import sys
import os
import json
import re
import torch
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Hugging Face 相關套件
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

# 本專案的模組
from common.utils import load_json
from common.tool_schema import get_tool_names


# ==============================================================================
# 設定參數
# ==============================================================================

# 模型設定
MODEL_NAME = "Qwen/Qwen3.5-2B"

# LoRA 設定
LORA_CONFIG = {
    "r": 8,                    # LoRA rank
    "lora_alpha": 8,           # LoRA alpha
    "lora_dropout": 0.05,       # Dropout
    "target_modules": [         # 要套用 LoRA 的層
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM"
}

# GRPO 訓練設定
GRPO_CONFIG = {
    "output_dir": "./grpo_output",
    "num_train_epochs": 2,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 2e-6,
    "logging_steps": 1,
    "save_steps": 8,
    "max_grad_norm": 1.0,
    "warmup_ratio": 0.07,
    "bf16": True,               # 使用 bfloat16（需要 GPU 支援）
    "gradient_checkpointing": True,  # 節省記憶體
    
    # GRPO 特定參數
    "beta": 0.01,
    "num_generations": 2,       # 每個 prompt 生成幾個回答

    "max_completion_length": 1024,      # 最大生成 token 數
    "temperature": 0.6,         # 生成溫度

}


# ==============================================================================
# Reward Function
# ==============================================================================

def extract_json_from_text(text: str) -> dict | None:
    """從文字中提取 JSON"""
    if not text:
        return None
    
    text = text.strip()
    
    # 嘗試直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # 嘗試從 markdown code block 提取
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(code_block_pattern, text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass
    
    # 尋找 JSON 物件
    start_idx = text.find('{')
    end_idx = text.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            return json.loads(text[start_idx:end_idx + 1])
        except json.JSONDecodeError:
            pass
    
    return None


def compute_reward(response: str, prompt_data: dict) -> float:
    """
    計算單一回應的 reward
    
    這個函式整合了 Lab 2 的 reward functions。
    
    Args:
        response: 模型生成的回應
        prompt_data: prompt 的 metadata（包含期望的工具等資訊）
    
    Returns:
        0.0 ~ 1.0 的 reward 分數
    """
    # TODO
    print(f"   📝 評分回應：{response}")
    response = response.strip().split("</think>")[-1].strip()  # 移除思考過程，專注於最終輸出
    
    # 特殊情況：應該追問
    if prompt_data.get("should_clarify", False):
        parsed = extract_json_from_text(response)
        if parsed is None:
            return 1.0  # 正確地選擇追問
        else:
            return 0.2  # 不應該輸出 JSON
    
    # 一般情況：檢查 JSON 格式和工具正確性
    parsed = extract_json_from_text(response)
    
    if parsed is None:
        return 0.0  # 無法解析 JSON
    
    if not isinstance(parsed, dict):
        return 0.1
    
    # 格式分數
    format_score = 0.3  # 基礎分：是 JSON 物件
    
    if "name" in parsed:
        format_score = 0.5
        
        if "arguments" in parsed:
            format_score = 0.7
            
            valid_tools = get_tool_names()
            if parsed["name"] in valid_tools:
                format_score = 0.85
                
                if parsed.get("type") == "tool_call":
                    format_score = 1.0
    
    # 工具正確性分數
    tool_score = 0.0
    expected_tool = prompt_data.get("expected_tool")
    
    if expected_tool:
        actual_tool = parsed.get("name")
        if actual_tool == expected_tool:
            tool_score = 0.5
            
            # 檢查參數
            metadata = prompt_data.get("metadata", {})
            actual_args = parsed.get("arguments", {})
            
            # 簡單檢查：參數中是否包含正確的值
            args_correct = True
            for key, expected_value in metadata.items():
                if key in ["order_id", "tracking_no", "reason"]:
                    if actual_args.get(key) != expected_value:
                        # 部分匹配也給一些分數
                        if expected_value in str(actual_args.get(key, "")):
                            tool_score = 0.7
                        args_correct = False
            
            if args_correct and actual_args:
                tool_score = 1.0
    
    # 組合分數
    total_reward = 0.4 * format_score + 0.6 * tool_score
    print("   格式分數：{:.2f}，工具分數：{:.2f}，總分：{:.2f}".format(format_score, tool_score, total_reward))
    
    return total_reward



# ==============================================================================
# 資料準備
# ==============================================================================

def load_training_data(filepath: str) -> Dataset:
    """
    載入訓練資料並轉換為 Dataset 格式
    """
    data = load_json(filepath)
    
    # GRPOTrainer 需要 "prompt" 欄位
    dataset_dict = {
        "prompt": [item["prompt"] for item in data],
        "task_type": [item.get("task_type", "") for item in data],
        "expected_tool": [item.get("expected_tool", "") for item in data],
        "should_clarify": [item.get("should_clarify", False) for item in data],
        "metadata": [json.dumps(item.get("metadata", {})) for item in data],
    }
    
    return Dataset.from_dict(dataset_dict)


# ==============================================================================
# 模型載入
# ==============================================================================

def load_model_and_tokenizer():
    """
    載入模型和 tokenizer，並設定 LoRA
    """
    print(f"📦 載入模型：{MODEL_NAME}")
    
    # 載入 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 設定 padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 量化設定（節省記憶體）
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # 載入模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 準備模型進行 k-bit 訓練
    model = prepare_model_for_kbit_training(model)
    
    # 套用 LoRA
    print("🔧 套用 LoRA 設定...")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    # 顯示可訓練參數
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   可訓練參數：{trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model, tokenizer


# ==============================================================================
# 主程式
# ==============================================================================

def main():
    """
    主程式
    """
    print("=" * 60)
    print("Lab 4：GRPO 訓練")
    print("=" * 60)
    
    # 檢查 GPU
    if not torch.cuda.is_available():
        print("\n❌ 未偵測到 GPU，GRPO 訓練需要 GPU")
        print("   請確認 CUDA 環境已正確設定")
        return
    
    print(f"\n🖥️  GPU：{torch.cuda.get_device_name(0)}")
    print(f"   VRAM：{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Step 1：載入訓練資料
    print("\n📂 載入訓練資料...")
    training_data_path = "../lab3/training_prompts.json"
    
    if not os.path.exists(training_data_path):
        print(f"   ❌ 找不到 {training_data_path}")
        print("   請先至 lab3 執行 python 1_prepare_dataset.py")
        return
    
    dataset = load_training_data(training_data_path)
    print(f"   載入 {len(dataset)} 筆訓練資料")
    
    # Step 2：載入模型
    print("\n🤖 載入模型...")
    model, tokenizer = load_model_and_tokenizer()
    
    # Step 3：設定 GRPO Trainer
    print("\n⚙️  設定 GRPO Trainer...")
    
    grpo_config = GRPOConfig(
        **GRPO_CONFIG,
        remove_unused_columns=False,  # 保留額外欄位
    )
    
    # 自定義 reward function（TRL 會傳入 completion_ids、trainer_state 等額外參數）
    def reward_fn(
        prompts,
        completions,
        task_type=None,
        expected_tool=None,
        should_clarify=None,
        metadata=None,
        completion_ids=None,
        trainer_state=None,
        **kwargs,
    ):
        """包裝 reward function，接收資料集欄位"""
        rewards = []
        for i, completion in enumerate(completions):
            prompt_data = {
                "task_type": task_type[i] if task_type else "",
                "expected_tool": expected_tool[i] if expected_tool else "",
                "should_clarify": should_clarify[i] if should_clarify else False,
                "metadata": json.loads(metadata[i]) if metadata else {},
            }
            reward = compute_reward(completion, prompt_data)
            rewards.append(reward)
        return rewards
    
    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=reward_fn,
    )
    
    # Step 4：開始訓練
    print("\n🚀 開始 GRPO 訓練...")
    print("-" * 60)
    
    trainer.train()
    
    print("-" * 60)
    print("\n✅ 訓練完成！")
    
    # Step 5：儲存模型
    print("\n💾 儲存模型...")
    output_dir = GRPO_CONFIG["output_dir"]
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print(f"   模型已儲存至 {output_dir}/final")
    
    print("\n✅ 全部完成！")
    print("   下一步：前往 lab5 執行 python 1_evaluate_trained.py 評估訓練成果")


if __name__ == "__main__":
    main()
