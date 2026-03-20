"""
==============================================================================
Lab 2：Reward Function 實作（解答版）
==============================================================================

這是 1_reward_functions.py 的解答版本。
學生應該先嘗試自己實作，再參考此解答。

注意：這個檔案被 2_test_rewards.py 引用，用於測試。
"""

import sys
import os
import json
import re
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.tool_schema import TOOLS, get_tool_names


def extract_json(response: str) -> Tuple[Optional[dict], str]:
    """
    從回應中提取 JSON 物件
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
    start_idx = response.find('{')
    end_idx = response.rfind('}')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            json_str = response[start_idx:end_idx + 1]
            return json.loads(json_str), ""
        except json.JSONDecodeError:
            return None, "JSON 語法錯誤"
    
    return None, "找不到 JSON 物件"


def format_reward(response: str) -> float:
    """
    評估回應的 JSON 格式正確性
    """
    parsed, error = extract_json(response)
    
    if parsed is None:
        return 0.0
    
    if not isinstance(parsed, dict):
        return 0.1
    
    score = 0.3
    
    if "name" in parsed:
        score = 0.5
        
        if "arguments" in parsed:
            score = 0.7
            
            valid_tools = get_tool_names()
            if parsed["name"] in valid_tools:
                score = 0.85
                
                if parsed.get("type") == "tool_call":
                    score = 1.0
    
    return score


def tool_correctness_reward(
    response: str, 
    expected_tool: str, 
    expected_args: dict
) -> float:
    """
    評估工具選擇和參數的正確性
    """
    parsed, error = extract_json(response)
    
    if parsed is None:
        return 0.0
    
    actual_tool = parsed.get("name")
    
    if actual_tool != expected_tool:
        return 0.3
    
    actual_args = parsed.get("arguments", {})
    
    all_args_correct = True
    for key, expected_value in expected_args.items():
        actual_value = actual_args.get(key)
        if actual_value != expected_value:
            all_args_correct = False
            break
    
    if not all_args_correct:
        return 0.6
    
    extra_args = set(actual_args.keys()) - set(expected_args.keys())
    if extra_args:
        return 0.8
    
    return 1.0


def clarification_reward(response: str) -> float:
    """
    評估模型是否正確地選擇追問
    """
    parsed, error = extract_json(response)
    
    if parsed is None:
        return 1.0
    
    arguments = parsed.get("arguments", {})
    
    if not arguments:
        return 0.5
    
    return 0.0


def combined_reward(
    response: str,
    expected_tool: Optional[str] = None,
    expected_args: Optional[dict] = None,
    should_clarify: bool = False,
    weights: dict = None
) -> dict:
    """
    組合多個 reward function 計算總分
    """
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
    
    if should_clarify:
        result["clarification"] = clarification_reward(response)
        result["total"] = result["clarification"]
        result["breakdown"] = f"clarification={result['clarification']:.2f}"
        return result
    
    result["format"] = format_reward(response)
    
    if expected_tool and expected_args is not None:
        result["tool_correctness"] = tool_correctness_reward(
            response, expected_tool, expected_args
        )
    
    total = 0.0
    breakdown_parts = []
    
    if "format" in weights:
        total += weights["format"] * result["format"]
        breakdown_parts.append(f"format={result['format']:.2f}")
    
    if "tool_correctness" in weights and expected_tool:
        total += weights["tool_correctness"] * result["tool_correctness"]
        breakdown_parts.append(f"tool={result['tool_correctness']:.2f}")
    
    if not expected_tool:
        total = result["format"]
    
    result["total"] = total
    result["breakdown"] = ", ".join(breakdown_parts)
    
    return result
