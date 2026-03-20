"""
==============================================================================
Lab 2：Reward Function 實作
==============================================================================

📌 本檔案功能：
    實作用於 GRPO 訓練的 Reward Function，包括：
    1. 格式 Reward：檢查 JSON 格式正確性
    2. 工具正確性 Reward：檢查工具選擇和參數
    3. 組合 Reward：將多個 reward 加權組合

📖 設計原則：
    - Reward 應該是連續的（有層次），而非只有 0/1
    - 部分正確應該得到部分分數
    - 越接近目標，分數越高

🔧 使用方式：
    from lab2.reward_functions import format_reward, tool_correctness_reward
    
    score = format_reward('{"type": "tool_call", ...}')
    print(f"格式分數：{score}")
"""

import sys
import os
import json
import re
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.tool_schema import TOOLS, get_tool_names


# ==============================================================================
# 輔助函式
# ==============================================================================

def extract_json(response: str) -> Tuple[Optional[dict], str]:
    """
    從回應中提取 JSON 物件
    
    嘗試多種方式解析：
    1. 直接解析整個字串
    2. 從 markdown code block 提取
    3. 尋找 JSON 物件模式
    
    Args:
        response: 模型的原始回應
    
    Returns:
        (parsed_json, error_message)
        - 成功：(dict, "")
        - 失敗：(None, "錯誤說明")
    """
    if not response or not response.strip():
        return None, "空回應"
    
    response = response.strip()
    
    # 方法 1：直接解析
    try:
        return json.loads(response), ""
    except json.JSONDecodeError:
        pass
    
    # 方法 2：從 markdown code block 提取
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    match = re.search(code_block_pattern, response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip()), ""
        except json.JSONDecodeError:
            pass
    
    # 方法 3：尋找 JSON 物件
    # 找到第一個 { 到最後一個 } 之間的內容
    start_idx = response.find('{')
    end_idx = response.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            json_str = response[start_idx:end_idx + 1]
            return json.loads(json_str), ""
        except json.JSONDecodeError:
            return None, "JSON 語法錯誤"
    
    return None, "找不到 JSON 物件"


# ==============================================================================
# Reward Function 1：格式檢查
# ==============================================================================

def format_reward(response: str) -> float:
    """
    評估回應的 JSON 格式正確性
    
    這個 reward function 檢查模型輸出是否符合 tool_call 的格式規範。
    使用層次化評分，讓模型能逐步學習正確格式。
    
    評分標準（從低到高）：
    ┌────────┬─────────────────────────────────────────┐
    │  分數  │ 條件                                    │
    ├────────┼─────────────────────────────────────────┤
    │  0.0   │ 無法解析為 JSON                         │
    │  0.3   │ 是 JSON 物件，但缺少關鍵欄位            │
    │  0.5   │ 有 name 欄位                            │
    │  0.7   │ 有 name + arguments 欄位               │
    │  0.85  │ name 是有效的工具名稱                   │
    │  1.0   │ 完全符合格式（含 type: tool_call）      │
    └────────┴─────────────────────────────────────────┘
    
    Args:
        response: 模型的原始回應字串
    
    Returns:
        0.0 ~ 1.0 的分數
    
    Example:
        >>> format_reward('{"type":"tool_call","name":"get_order_status","arguments":{}}')
        1.0
        >>> format_reward('{"name":"get_order_status"}')
        0.5
        >>> format_reward('這不是 JSON')
        0.0
    """
    
    # TODO: 實作格式 reward function
    # 
    # Step 1: 使用 extract_json() 嘗試解析 JSON
    # Step 2: 根據解析結果和欄位存在與否給分
    # 
    # 提示：
    # - parsed, error = extract_json(response)
    # - 如果 parsed 是 None，回傳 0.0
    # - 檢查是否有 "name" 欄位
    # - 檢查是否有 "arguments" 欄位
    # - 檢查 name 是否在 get_tool_names() 中
    # - 檢查是否有 "type": "tool_call"
    
    # Step 1：嘗試解析 JSON
    parsed, error = extract_json(response)
    
    if parsed is None:
        # 無法解析為 JSON
        return 0.0
    
    # Step 2：檢查是否為字典
    if not isinstance(parsed, dict):
        return 0.1
    
    # 基礎分數：至少是 JSON 物件
    score = 0.3
    
    # Step 3：檢查 name 欄位
    if "name" in parsed:
        score = 0.5
        
        # Step 4：檢查 arguments 欄位
        if "arguments" in parsed:
            score = 0.7
            
            # Step 5：檢查 name 是否為有效工具
            valid_tools = get_tool_names()
            if parsed["name"] in valid_tools:
                score = 0.85
                
                # Step 6：檢查是否有 type: tool_call
                if parsed.get("type") == "tool_call":
                    score = 1.0
    
    return score


# ==============================================================================
# Reward Function 2：工具正確性
# ==============================================================================

def tool_correctness_reward(
    response: str, 
    expected_tool: str, 
    expected_args: dict
) -> float:
    """
    評估工具選擇和參數的正確性
    
    這個 reward function 檢查模型是否選擇了正確的工具，
    並填入了正確的參數。
    
    評分標準：
    ┌────────┬─────────────────────────────────────────┐
    │  分數  │ 條件                                    │
    ├────────┼─────────────────────────────────────────┤
    │  0.0   │ JSON 解析失敗                           │
    │  0.3   │ JSON 正確但工具名稱錯誤                 │
    │  0.6   │ 工具名稱正確但參數錯誤/缺失             │
    │  0.8   │ 工具和必填參數正確，但有額外參數        │
    │  1.0   │ 工具和參數都完全正確                    │
    └────────┴─────────────────────────────────────────┘
    
    Args:
        response: 模型的原始回應字串
        expected_tool: 期望的工具名稱
        expected_args: 期望的參數字典
    
    Returns:
        0.0 ~ 1.0 的分數
    
    Example:
        >>> tool_correctness_reward(
        ...     '{"name":"get_order_status","arguments":{"order_id":"A123456789"}}',
        ...     "get_order_status",
        ...     {"order_id": "A123456789"}
        ... )
        1.0
    """
    
    # TODO: 實作工具正確性 reward function
    #
    # Step 1: 解析 JSON
    # Step 2: 檢查工具名稱是否正確
    # Step 3: 檢查參數是否正確
    #
    # 提示：
    # - 使用 extract_json() 解析
    # - 比較 parsed.get("name") 和 expected_tool
    # - 比較 parsed.get("arguments", {}) 和 expected_args
    
    # Step 1：解析 JSON
    parsed, error = extract_json(response)
    
    if parsed is None:
        return 0.0
    
    # Step 2：檢查工具名稱
    actual_tool = parsed.get("name")
    
    if actual_tool != expected_tool:
        # 工具名稱錯誤，但至少是 JSON
        return 0.3
    
    # Step 3：檢查參數
    actual_args = parsed.get("arguments", {})
    
    # 檢查所有期望的參數是否存在且正確
    all_args_correct = True
    for key, expected_value in expected_args.items():
        actual_value = actual_args.get(key)
        if actual_value != expected_value:
            all_args_correct = False
            break
    
    if not all_args_correct:
        # 工具正確但參數錯誤
        return 0.6
    
    # 檢查是否有額外的參數
    extra_args = set(actual_args.keys()) - set(expected_args.keys())
    if extra_args:
        # 有額外參數（不一定是錯的，但不完美）
        return 0.8
    
    # 完全正確
    return 1.0


# ==============================================================================
# Reward Function 3：追問檢測
# ==============================================================================

def clarification_reward(response: str) -> float:
    """
    評估模型是否正確地選擇追問（而非亂猜）
    
    當資訊不足時，好的助理應該追問，而非輸出不完整的 tool call。
    
    評分標準：
    - 1.0：沒有輸出 JSON（正確選擇追問）
    - 0.5：輸出了 JSON 但缺少必填參數（部分正確）
    - 0.0：輸出了完整 JSON（不應該猜測）
    
    Args:
        response: 模型的原始回應字串
    
    Returns:
        0.0 ~ 1.0 的分數
    
    Example:
        >>> clarification_reward("請問您的訂單編號是什麼？")
        1.0
        >>> clarification_reward('{"name":"get_order_status","arguments":{}}')
        0.5
    """
    
    parsed, error = extract_json(response)
    
    if parsed is None:
        # 沒有輸出 JSON，正確選擇追問
        return 1.0
    
    # 檢查是否有填入參數
    arguments = parsed.get("arguments", {})
    
    if not arguments:
        # 有 JSON 但沒參數，部分正確（可能知道要用什麼工具）
        return 0.5
    
    # 輸出了完整的 JSON，但資訊不足時不應該這樣
    return 0.0


# ==============================================================================
# 組合 Reward Function
# ==============================================================================

def combined_reward(
    response: str,
    expected_tool: Optional[str] = None,
    expected_args: Optional[dict] = None,
    should_clarify: bool = False,
    weights: dict = None
) -> dict:
    """
    組合多個 reward function 計算總分
    
    根據不同場景使用不同的 reward 組合：
    - 一般 tool call 場景：format + tool_correctness
    - 需要追問的場景：clarification
    
    Args:
        response: 模型的原始回應
        expected_tool: 期望的工具名稱（可選）
        expected_args: 期望的參數（可選）
        should_clarify: 是否為需要追問的場景
        weights: 各 reward 的權重（可選）
    
    Returns:
        包含各項分數和總分的字典
        {
            "format": 0.8,
            "tool_correctness": 0.6,
            "total": 0.7,
            "breakdown": "format=0.8, tool=0.6"
        }
    
    Example:
        >>> result = combined_reward(
        ...     '{"name":"get_order_status","arguments":{"order_id":"A123"}}',
        ...     expected_tool="get_order_status",
        ...     expected_args={"order_id": "A123456789"}
        ... )
        >>> print(result["total"])
        0.7
    """
    
    # 預設權重
    if weights is None:
        weights = {
            "format": 0.4,
            "tool_correctness": 0.6
        }
    
    result = {
        "format": 0.0,
        "tool_correctness": 0.0,
        "clarification": 0.0,
        "total": 0.0,
        "breakdown": ""
    }
    
    # 場景 1：需要追問
    if should_clarify:
        result["clarification"] = clarification_reward(response)
        result["total"] = result["clarification"]
        result["breakdown"] = f"clarification={result['clarification']:.2f}"
        return result
    
    # 場景 2：一般 tool call
    result["format"] = format_reward(response)
    
    if expected_tool and expected_args is not None:
        result["tool_correctness"] = tool_correctness_reward(
            response, expected_tool, expected_args
        )
    
    # 計算加權總分
    total = 0.0
    breakdown_parts = []
    
    if "format" in weights:
        total += weights["format"] * result["format"]
        breakdown_parts.append(f"format={result['format']:.2f}")
    
    if "tool_correctness" in weights and expected_tool:
        total += weights["tool_correctness"] * result["tool_correctness"]
        breakdown_parts.append(f"tool={result['tool_correctness']:.2f}")
    
    # 如果沒有 tool_correctness，調整權重
    if not expected_tool:
        total = result["format"]
    
    result["total"] = total
    result["breakdown"] = ", ".join(breakdown_parts)
    
    return result


# ==============================================================================
# 主程式（測試用）
# ==============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Reward Function 測試")
    print("=" * 60)
    
    # 測試案例
    test_cases = [
        # 完美格式
        {
            "response": '{"type":"tool_call","name":"get_order_status","arguments":{"order_id":"A123456789"}}',
            "expected_tool": "get_order_status",
            "expected_args": {"order_id": "A123456789"}
        },
        # 缺少 type 欄位
        {
            "response": '{"name":"get_order_status","arguments":{"order_id":"A123456789"}}',
            "expected_tool": "get_order_status",
            "expected_args": {"order_id": "A123456789"}
        },
        # 工具錯誤
        {
            "response": '{"name":"wrong_tool","arguments":{"order_id":"A123456789"}}',
            "expected_tool": "get_order_status",
            "expected_args": {"order_id": "A123456789"}
        },
        # 參數錯誤
        {
            "response": '{"name":"get_order_status","arguments":{"order_id":"wrong"}}',
            "expected_tool": "get_order_status",
            "expected_args": {"order_id": "A123456789"}
        },
        # 非 JSON
        {
            "response": "好的，我來幫你查詢訂單狀態。",
            "expected_tool": "get_order_status",
            "expected_args": {"order_id": "A123456789"}
        },
    ]
    
    for i, case in enumerate(test_cases):
        print(f"\n--- 測試案例 {i+1} ---")
        print(f"回應：{case['response'][:60]}...")
        
        # 測試個別 reward
        fmt_score = format_reward(case["response"])
        tool_score = tool_correctness_reward(
            case["response"],
            case["expected_tool"],
            case["expected_args"]
        )
        
        print(f"格式分數：{fmt_score:.2f}")
        print(f"工具分數：{tool_score:.2f}")
        
        # 測試組合 reward
        combined = combined_reward(
            case["response"],
            case["expected_tool"],
            case["expected_args"]
        )
        print(f"總分：{combined['total']:.2f} ({combined['breakdown']})")
