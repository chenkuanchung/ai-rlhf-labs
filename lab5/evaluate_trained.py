"""
==============================================================================
Lab 5：評估訓練後的模型
==============================================================================

📌 本檔案功能：
    1. 載入訓練前（baseline）和訓練後的模型
    2. 對相同的測試案例進行推論
    3. 比較兩者的表現差異

📖 評估重點：
    - JSON 格式正確率是否提升
    - 工具選擇準確度是否提升
    - 參數填寫正確率是否提升

🔧 使用方式：
    cd lab5
    python 1_evaluate_trained.py
"""

import sys
import os
import json
import re
import torch
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from common.utils import load_jsonl, save_jsonl
from common.prompts import system_prompt
from common.tool_schema import get_tool_names


# ==============================================================================
# 設定
# ==============================================================================

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
TRAINED_MODEL_PATH = "../lab4/grpo_output/final"

# 測試案例
TEST_CASES = [
    {
        "user_input": "幫我查訂單 A123456789 目前狀態",
        "expected_tool": "get_order_status",
        "metadata": {"order_id": "A123456789"},
        "should_clarify": False,
    },
    {
        "user_input": "我要查物流 TWD12345678 到哪了",
        "expected_tool": "track_shipment",
        "metadata": {"tracking_no": "TWD12345678"},
        "should_clarify": False,
    },
    {
        "user_input": "我要退款",
        "expected_tool": None,
        "metadata": {},
        "should_clarify": True,
    },
    {
        "user_input": "查一下 A999888777 這筆訂單",
        "expected_tool": "get_order_status",
        "metadata": {"order_id": "A999888777"},
        "should_clarify": False,
    },
    {
        "user_input": "物流單號 TWD88888888，幫我追蹤一下",
        "expected_tool": "track_shipment",
        "metadata": {"tracking_no": "TWD88888888"},
        "should_clarify": False,
    },
]


# ==============================================================================
# Reward 計算（與 Lab 4 訓練時一致）
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
    
    與 Lab 4 訓練時使用的 reward function 一致。
    
    Args:
        response: 模型生成的回應
        prompt_data: prompt 的 metadata（包含期望的工具等資訊）
    
    Returns:
        0.0 ~ 1.0 的 reward 分數
    """
    
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
    
    return total_reward


# ==============================================================================
# 模型載入與推論
# ==============================================================================

def format_prompt(user_input: str) -> str:
    """格式化 prompt"""
    sys_prompt = system_prompt()
    return f"""<|im_start|>system
{sys_prompt}<|im_end|>
<|im_start|>user
{user_input}<|im_end|>
<|im_start|>assistant
"""


def load_base_model():
    """載入基礎模型（訓練前）"""
    print("📦 載入基礎模型...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    return model, tokenizer


def load_trained_model(base_model):
    """載入訓練後的模型（LoRA adapter）"""
    print("📦 載入訓練後的模型...")
    
    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"   ❌ 找不到訓練後的模型：{TRAINED_MODEL_PATH}")
        return None
    
    model = PeftModel.from_pretrained(base_model, TRAINED_MODEL_PATH)
    return model


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256) -> str:
    """生成回應"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取 assistant 回應部分
    assistant_marker = "<|im_start|>assistant\n"
    if assistant_marker in response:
        response = response.split(assistant_marker)[-1]
    
    # 移除結束標記
    end_marker = "<|im_end|>"
    if end_marker in response:
        response = response.split(end_marker)[0]
    
    return response.strip()


def evaluate_model(model, tokenizer, test_cases: List[Dict], model_name: str) -> Dict:
    """評估模型表現"""
    print(f"\n🔍 評估 {model_name}...")
    
    results = []
    total_reward = 0
    json_valid_count = 0
    tool_correct_count = 0
    
    for i, case in enumerate(test_cases):
        prompt = format_prompt(case["user_input"])
        response = generate_response(model, tokenizer, prompt)
        
        # 計算 reward
        prompt_data = {
            "expected_tool": case["expected_tool"],
            "should_clarify": case["should_clarify"],
            "metadata": case["metadata"],
        }
        reward = compute_reward(response, prompt_data)
        total_reward += reward
        
        # 檢查 JSON 有效性
        parsed = extract_json_from_text(response)
        if case["should_clarify"]:
            # 追問場景：沒有 JSON 是正確的
            json_valid = parsed is None
            tool_correct = parsed is None
        else:
            json_valid = parsed is not None
            tool_correct = json_valid and parsed.get("name") == case["expected_tool"]
        
        if json_valid:
            json_valid_count += 1
        if tool_correct:
            tool_correct_count += 1
        
        result = {
            "user_input": case["user_input"],
            "response": response[:200] + "..." if len(response) > 200 else response,
            "reward": reward,
            "json_valid": json_valid,
            "tool_correct": tool_correct,
        }
        results.append(result)
        
        print(f"   [{i+1}/{len(test_cases)}] reward={reward:.2f} | JSON={'✓' if json_valid else '✗'} | Tool={'✓' if tool_correct else '✗'}")
    
    metrics = {
        "model_name": model_name,
        "avg_reward": total_reward / len(test_cases),
        "json_valid_rate": json_valid_count / len(test_cases),
        "tool_correct_rate": tool_correct_count / len(test_cases),
        "results": results,
    }
    
    return metrics


def print_comparison(baseline_metrics: Dict, trained_metrics: Dict):
    """列印比較結果"""
    print("\n" + "=" * 60)
    print("📊 訓練前後比較")
    print("=" * 60)
    
    metrics_names = [
        ("avg_reward", "平均 Reward"),
        ("json_valid_rate", "JSON 正確率"),
        ("tool_correct_rate", "工具正確率"),
    ]
    
    print(f"\n{'指標':<15} {'訓練前':>12} {'訓練後':>12} {'變化':>12}")
    print("-" * 55)
    
    for key, name in metrics_names:
        baseline_val = baseline_metrics[key]
        trained_val = trained_metrics[key]
        diff = trained_val - baseline_val
        
        # 顏色指示
        if diff > 0:
            arrow = "↑"
        elif diff < 0:
            arrow = "↓"
        else:
            arrow = "→"
        
        print(f"{name:<15} {baseline_val:>11.1%} {trained_val:>11.1%} {arrow} {abs(diff):>+10.1%}")
    
    print("\n" + "-" * 60)
    
    # 顯示詳細對比
    print("\n📋 詳細對比（每個測試案例）：")
    print("-" * 60)
    
    for i, (base_r, train_r) in enumerate(zip(
        baseline_metrics["results"], 
        trained_metrics["results"]
    )):
        print(f"\n案例 {i+1}：{base_r['user_input'][:40]}...")
        print(f"   訓練前 reward: {base_r['reward']:.2f}")
        print(f"   訓練後 reward: {train_r['reward']:.2f}")
        
        if train_r['reward'] > base_r['reward']:
            print(f"   → 改善 +{train_r['reward'] - base_r['reward']:.2f}")
        elif train_r['reward'] < base_r['reward']:
            print(f"   → 退步 {train_r['reward'] - base_r['reward']:.2f}")
        else:
            print(f"   → 持平")


def main():
    """主程式"""
    print("=" * 60)
    print("Lab 5：評估訓練成果")
    print("=" * 60)
    
    # 檢查 GPU
    if not torch.cuda.is_available():
        print("\n❌ 未偵測到 GPU")
        return
    
    # 載入基礎模型
    base_model, tokenizer = load_base_model()
    
    # 評估基礎模型
    baseline_metrics = evaluate_model(base_model, tokenizer, TEST_CASES, "基礎模型（訓練前）")
    
    # 載入訓練後的模型
    trained_model = load_trained_model(base_model)
    
    if trained_model is None:
        print("\n⚠️  跳過訓練後模型的評估")
        print("   請先至 lab4 執行 python 1_grpo_training.py 進行訓練")
        
        # 只顯示基礎模型的結果
        print("\n📊 基礎模型表現：")
        print(f"   平均 Reward：{baseline_metrics['avg_reward']:.1%}")
        print(f"   JSON 正確率：{baseline_metrics['json_valid_rate']:.1%}")
        print(f"   工具正確率：{baseline_metrics['tool_correct_rate']:.1%}")
        return
    
    # 評估訓練後的模型
    trained_metrics = evaluate_model(trained_model, tokenizer, TEST_CASES, "訓練後模型")
    
    # 顯示比較結果
    print_comparison(baseline_metrics, trained_metrics)
    
    # 儲存結果
    output = {
        "baseline": baseline_metrics,
        "trained": trained_metrics,
    }
    
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    
    print("\n💾 評估結果已儲存至 evaluation_results.json")
    print("\n✅ 評估完成！")


if __name__ == "__main__":
    main()
