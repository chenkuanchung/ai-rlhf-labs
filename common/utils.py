"""
==============================================================================
Common 模組：輔助工具函式 (utils.py)
==============================================================================

📌 本檔案功能：
    提供專案中常用的輔助函式，包括：
    1. JSON 格式化輸出
    2. 其他共用工具

🔧 使用方式：
    from common.utils import pretty
    
    data = {"name": "test", "value": 123}
    print(pretty(data))
"""

import json
import re
from typing import Any, Dict, Optional

def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    """
    從文字中提取第一個 JSON 物件
    
    LLM 的輸出可能包含 JSON 和自然語言混合的內容，
    這個函式會嘗試從中提取出 JSON 物件。
    
    Args:
        text: 可能包含 JSON 的文字
    
    Returns:
        解析成功的 JSON 物件（dict），或 None
    
    提取策略：
    ┌─────────────────────────────────────────────────────────────────┐
    │  策略 1：尋找 ```json ... ``` 區塊                               │
    │    適用於 LLM 用 markdown code block 包裝的情況                  │
    │                                                                 │
    │  策略 2：尋找 {...} 並用括號匹配                                 │
    │    適用於 LLM 直接輸出 JSON 的情況                               │
    └─────────────────────────────────────────────────────────────────┘
    
    Example:
        # 從 markdown code block 提取
        >>> text = '```json\n{"name": "test"}\n```'
        >>> extract_json_block(text)
        {"name": "test"}
        
        # 從純文字提取
        >>> text = '這是 JSON: {"name": "test"} 結束'
        >>> extract_json_block(text)
        {"name": "test"}
        
        # 無法提取
        >>> extract_json_block("沒有 JSON")
        None
    
    注意事項：
        - 只提取第一個 JSON 物件（dict，不是 list）
        - 巢狀的 JSON 也可以處理
        - 如果 JSON 格式錯誤，回傳 None
    """
    
    # 空值檢查
    if not text:
        return None
    
    # ==========================================================================
    # 策略 1：尋找 ```json ... ``` 區塊
    # ==========================================================================
    # 這是 LLM 常用的格式，用 markdown code block 包裝 JSON
    # 例如：
    # ```json
    # {"type": "tool_call", "name": "get_order_status", ...}
    # ```
    
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            # JSON 格式錯誤，繼續嘗試其他策略
            pass
    
    # ==========================================================================
    # 策略 2：括號匹配（Brace Matching）
    # ==========================================================================
    # 找到第一個 {，然後匹配到對應的 }
    # 這可以處理巢狀的 JSON
    
    start = text.find("{")
    if start == -1:
        # 沒有找到 {
        return None
    
    # 用深度計數來找到配對的 }
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                # 找到配對的 }
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # 不是合法的 JSON
                    return None
    
    # 沒有找到配對的 }
    return None



def pretty(obj: Any, indent: int = 2) -> str:
    """
    將物件格式化為漂亮的 JSON 字串
    
    Args:
        obj: 要格式化的物件
        indent: 縮排空格數（預設 2）
    
    Returns:
        格式化的 JSON 字串
    
    Example:
        >>> data = {"name": "test", "values": [1, 2, 3]}
        >>> print(pretty(data))
        {
          "name": "test",
          "values": [
            1,
            2,
            3
          ]
        }
    """
    return json.dumps(obj, ensure_ascii=False, indent=indent)


def load_json(filepath: str) -> list[dict]:
    """
    讀取 JSON 檔案
    
    Args:
        filepath: 檔案路徑
    
    Returns:
        JSON 物件（通常是列表）
    
    Example:
        >>> cases = load_json("eval_cases.json")
        >>> print(len(cases))
        5
    """
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: list[dict], filepath: str) -> None:
    """
    儲存資料為 JSON 檔案
    
    Args:
        data: JSON 物件（通常是列表）
        filepath: 輸出檔案路徑
    
    Example:
        >>> outputs = [{"id": "1", "response": "hello"}]
        >>> save_json(outputs, "outputs.json")
    """
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
