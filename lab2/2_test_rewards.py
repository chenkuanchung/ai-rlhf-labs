"""
==============================================================================
Lab 2：測試 Reward Function
==============================================================================

📌 本程式功能：
    1. 載入範例 completions
    2. 對每個 completion 計算 reward
    3. 展示 reward 分數的分布
    4. 幫助理解 GRPO 訓練的資料準備流程

📖 學習目標：
    - 理解 reward function 如何區分好壞輸出
    - 觀察不同類型錯誤的分數差異
    - 預覽 GRPO 訓練資料的格式

🔧 使用方式：
    cd lab2
    python 2_test_rewards.py
"""

import sys
import os
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lab2.reward_functions_solution import (
    format_reward,
    tool_correctness_reward,
    combined_reward,
    clarification_reward
)
from common.utils import load_json, save_json


def create_sample_completions():
    """
    建立範例 completions 資料
    
    模擬模型對同一個 prompt 生成多個不同的回應，
    展示不同品質的輸出會得到不同的 reward。
    """
    
    samples = [
        # ==========================================================================
        # 案例 1：查詢訂單狀態
        # ==========================================================================
        {
            "id": "sample_1",
            "prompt": "幫我查訂單 A123456789 目前狀態",
            "expected_tool": "get_order_status",
            "expected_args": {"order_id": "A123456789"},
            "should_clarify": False,
            "completions": [
                # 完美回應
                '{"type":"tool_call","name":"get_order_status","arguments":{"order_id":"A123456789"}}',
                # 缺少 type 欄位（輕微格式問題）
                '{"name":"get_order_status","arguments":{"order_id":"A123456789"}}',
                # 工具名稱錯誤
                '{"type":"tool_call","name":"track_shipment","arguments":{"tracking_no":"A123456789"}}',
                # 參數格式錯誤
                '{"type":"tool_call","name":"get_order_status","arguments":{"order":"A123456789"}}',
                # 直接用自然語言回答（沒有呼叫工具）
                '好的，我來幫你查詢訂單 A123456789 的狀態...',
                # JSON 語法錯誤
                '{"name":"get_order_status","arguments":{"order_id":"A123456789"',
            ]
        },
        
        # ==========================================================================
        # 案例 2：查詢物流
        # ==========================================================================
        {
            "id": "sample_2",
            "prompt": "我要查物流 TWD12345678 到哪了",
            "expected_tool": "track_shipment",
            "expected_args": {"tracking_no": "TWD12345678"},
            "should_clarify": False,
            "completions": [
                # 完美回應
                '{"type":"tool_call","name":"track_shipment","arguments":{"tracking_no":"TWD12345678"}}',
                # 參數名稱錯誤
                '{"type":"tool_call","name":"track_shipment","arguments":{"tracking":"TWD12345678"}}',
                # 選錯工具
                '{"type":"tool_call","name":"get_order_status","arguments":{"order_id":"TWD12345678"}}',
                # Markdown code block 包裝（應該也能解析）
                '```json\n{"type":"tool_call","name":"track_shipment","arguments":{"tracking_no":"TWD12345678"}}\n```',
            ]
        },
        
        # ==========================================================================
        # 案例 3：需要追問的情況
        # ==========================================================================
        {
            "id": "sample_3",
            "prompt": "我要退款",
            "expected_tool": None,
            "expected_args": None,
            "should_clarify": True,
            "completions": [
                # 正確：追問缺少的資訊
                '好的，請問您要退款的訂單編號是什麼？退款原因是什麼呢？',
                # 正確：另一種追問方式
                '我需要您的訂單編號和退款原因才能處理，請問訂單編號是？',
                # 錯誤：亂猜參數
                '{"type":"tool_call","name":"create_refund_request","arguments":{"order_id":"A000000000","reason":"其他"}}',
                # 部分錯誤：知道要用什麼工具但沒填參數
                '{"type":"tool_call","name":"create_refund_request","arguments":{}}',
            ]
        },
    ]
    
    return samples


def evaluate_completions(samples: list) -> list:
    """
    對所有 completions 計算 reward
    
    Args:
        samples: 範例資料列表
    
    Returns:
        包含 reward 分數的結果列表
    """
    
    results = []
    
    for sample in samples:
        sample_result = {
            "id": sample["id"],
            "prompt": sample["prompt"],
            "expected_tool": sample["expected_tool"],
            "should_clarify": sample["should_clarify"],
            "completions": []
        }
        
        for completion in sample["completions"]:
            # 計算 reward
            reward_result = combined_reward(
                response=completion,
                expected_tool=sample["expected_tool"],
                expected_args=sample["expected_args"],
                should_clarify=sample["should_clarify"]
            )
            
            sample_result["completions"].append({
                "text": completion,
                "reward": reward_result["total"],
                "format_score": reward_result["format"],
                "tool_score": reward_result["tool_correctness"],
                "clarification_score": reward_result["clarification"],
                "breakdown": reward_result["breakdown"]
            })
        
        results.append(sample_result)
    
    return results


def print_results(results: list) -> None:
    """
    美觀地列印評估結果
    """
    
    for sample in results:
        print("\n" + "=" * 70)
        print(f"📝 案例：{sample['id']}")
        print(f"   Prompt：{sample['prompt']}")
        
        if sample["should_clarify"]:
            print(f"   期望：模型應該追問（資訊不足）")
        else:
            print(f"   期望工具：{sample['expected_tool']}")
        
        print("-" * 70)
        print(f"{'Completion':<50} {'Reward':>10}")
        print("-" * 70)
        
        # 按 reward 排序
        sorted_completions = sorted(
            sample["completions"],
            key=lambda x: x["reward"],
            reverse=True
        )
        
        for i, comp in enumerate(sorted_completions):
            # 截斷過長的文字
            text = comp["text"][:45] + "..." if len(comp["text"]) > 45 else comp["text"]
            text = text.replace("\n", " ")
            
            # 用顏色標示（在終端機中）
            reward = comp["reward"]
            if reward >= 0.9:
                status = "🟢"
            elif reward >= 0.6:
                status = "🟡"
            else:
                status = "🔴"
            
            print(f"{status} {text:<47} {reward:>8.2f}")
        
        # 顯示分數統計
        rewards = [c["reward"] for c in sample["completions"]]
        print("-" * 70)
        print(f"   最高分：{max(rewards):.2f} | 最低分：{min(rewards):.2f} | 平均：{sum(rewards)/len(rewards):.2f}")


def create_grpo_format(results: list) -> list:
    """
    將結果轉換為 GRPO 訓練格式
    
    GRPO 需要的格式：
    {
        "prompt": "...",
        "completions": [
            {"text": "...", "reward": 0.9},
            {"text": "...", "reward": 0.3},
            ...
        ]
    }
    """
    
    grpo_data = []
    
    for sample in results:
        grpo_item = {
            "prompt": sample["prompt"],
            "completions": [
                {
                    "text": c["text"],
                    "reward": c["reward"]
                }
                for c in sample["completions"]
            ]
        }
        grpo_data.append(grpo_item)
    
    return grpo_data


def main():
    """
    主程式
    """
    print("=" * 70)
    print("Lab 2：Reward Function 測試")
    print("=" * 70)
    
    # Step 1：建立範例資料
    print("\n📦 建立範例 completions...")
    samples = create_sample_completions()
    print(f"   共 {len(samples)} 個案例")
    
    # Step 2：計算 reward
    print("\n🔢 計算 reward 分數...")
    results = evaluate_completions(samples)
    
    # Step 3：顯示結果
    print_results(results)
    
    # Step 4：產生 GRPO 格式資料
    print("\n\n" + "=" * 70)
    print("📋 GRPO 訓練資料格式預覽")
    print("=" * 70)
    
    grpo_data = create_grpo_format(results)
    
    # 顯示第一個範例
    print("\n範例資料結構：")
    print(json.dumps(grpo_data[0], ensure_ascii=False, indent=2)[:500] + "...")
    
    # 儲存 GRPO 格式資料
    output_file = "grpo_sample_data.json"
    save_json(grpo_data, output_file)
    print(f"\n💾 GRPO 格式資料已儲存至 {output_file}")
    
    # Step 5：統計摘要
    print("\n\n" + "=" * 70)
    print("📊 統計摘要")
    print("=" * 70)
    
    all_rewards = []
    for sample in results:
        all_rewards.extend([c["reward"] for c in sample["completions"]])
    
    high_reward = sum(1 for r in all_rewards if r >= 0.8)
    medium_reward = sum(1 for r in all_rewards if 0.4 <= r < 0.8)
    low_reward = sum(1 for r in all_rewards if r < 0.4)
    
    print(f"\n總 completion 數：{len(all_rewards)}")
    print(f"  🟢 高分 (>=0.8)：{high_reward} ({high_reward/len(all_rewards)*100:.1f}%)")
    print(f"  🟡 中分 (0.4-0.8)：{medium_reward} ({medium_reward/len(all_rewards)*100:.1f}%)")
    print(f"  🔴 低分 (<0.4)：{low_reward} ({low_reward/len(all_rewards)*100:.1f}%)")
    
    print("\n✅ 測試完成！")
    print("\n💡 觀察重點：")
    print("   1. 完美格式的回應得到最高分")
    print("   2. 部分正確（如缺少 type 欄位）得到中等分數")
    print("   3. 完全錯誤（非 JSON）得到最低分")
    print("   4. 在追問場景中，不輸出 JSON 反而是正確的")


if __name__ == "__main__":
    main()
