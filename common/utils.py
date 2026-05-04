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
from typing import Any


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
