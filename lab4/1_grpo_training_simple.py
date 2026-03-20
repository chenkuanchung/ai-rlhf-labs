"""
==============================================================================
Lab 4：GRPO 訓練（簡化版）
==============================================================================

📌 本檔案功能：
    這是一個簡化版的 GRPO 訓練範例，更容易理解核心概念。
    
    如果 1_grpo_training.py 太複雜或遇到問題，可以先看這個版本。

📖 核心流程：
    1. 準備 prompts
    2. 定義 reward function
    3. 設定 GRPOTrainer
    4. 開始訓練

🔧 使用方式：
    cd lab4
    python 1_grpo_training_simple.py
"""

import sys
import os
import json
import re
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model

from common.tool_schema import get_tool_names
from common.prompts import system_prompt


# ==============================================================================
# 簡化的設定
# ==============================================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# 簡單的訓練 prompts
TRAINING_PROMPTS = [
    "幫我查訂單 A123456789 目前狀態",
    "我的訂單 A111111111 到哪了",
    "查一下 A222222222 這筆訂單",
    "我要查物流 TWD12345678 到哪了",
    "物流單號 TWD88888888，幫我追蹤一下",
    "訂單編號 A333333333，幫我看一下狀態",
    "請問訂單 A444444444 現在是什麼情況",
    "查一下 TWD99999999 的配送進度",
]


# ==============================================================================
# Reward Function（簡化版）
# ==============================================================================

def simple_reward_function(completions, prompts, **kwargs):
    """
    簡化的 reward function
    
    只檢查基本的 JSON 格式和工具名稱。
    """
    rewards = []
    valid_tools = get_tool_names()
    
    for completion in completions:
        reward = 0.0
        
        # 嘗試解析 JSON
        try:
            # 找到 JSON 部分
            text = completion.strip()
            start = text.find('{')
            end = text.rfind('}')
            
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                parsed = json.loads(json_str)
                
                # 基礎分：是 JSON
                reward = 0.3
                
                # 有 name 欄位
                if "name" in parsed:
                    reward = 0.5
                    
                    # name 是有效工具
                    if parsed["name"] in valid_tools:
                        reward = 0.7
                        
                        # 有 arguments
                        if "arguments" in parsed:
                            reward = 0.85
                            
                            # 有 type: tool_call
                            if parsed.get("type") == "tool_call":
                                reward = 1.0
        except:
            pass
        
        rewards.append(reward)
    
    return rewards


# ==============================================================================
# 主程式
# ==============================================================================

def main():
    print("=" * 60)
    print("Lab 4：GRPO 訓練（簡化版）")
    print("=" * 60)
    
    # 檢查 GPU
    if not torch.cuda.is_available():
        print("\n❌ 需要 GPU 來執行訓練")
        print("   如果沒有 GPU，請閱讀程式碼了解流程")
        return
    
    print(f"\n🖥️  GPU：{torch.cuda.get_device_name(0)}")
    
    # Step 1：準備資料
    print("\n📝 準備訓練資料...")
    
    sys_prompt = system_prompt()
    
    # 格式化 prompts
    formatted_prompts = []
    for user_input in TRAINING_PROMPTS:
        prompt = f"""<|im_start|>system
{sys_prompt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""
        formatted_prompts.append(prompt)
    
    dataset = Dataset.from_dict({"prompt": formatted_prompts})
    print(f"   共 {len(dataset)} 筆訓練資料")
    
    # Step 2：載入模型
    print("\n📦 載入模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 套用 LoRA
    print("🔧 套用 LoRA...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Step 3：設定 GRPO
    print("\n⚙️  設定 GRPO Trainer...")
    
    config = GRPOConfig(
        output_dir="./grpo_simple_output",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        logging_steps=1,
        num_generations=2,      # 每個 prompt 生成 2 個回答
        max_new_tokens=128,
        temperature=0.7,
        kl_coef=0.05,
        bf16=True,
    )
    
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        config=config,
        train_dataset=dataset,
        reward_funcs=simple_reward_function,
    )
    
    # Step 4：訓練
    print("\n🚀 開始訓練...")
    print("-" * 60)
    print("觀察重點：")
    print("  - reward/mean：平均 reward，應該上升")
    print("  - kl：KL 散度，不要太大")
    print("-" * 60)
    
    trainer.train()
    
    print("\n✅ 訓練完成！")
    
    # 儲存模型
    trainer.save_model("./grpo_simple_output/final")
    print("💾 模型已儲存")


if __name__ == "__main__":
    main()
