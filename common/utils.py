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


def load_jsonl(filepath: str) -> list[dict]:
    """
    讀取 JSONL 檔案
    
    JSONL 格式：每行一個 JSON 物件
    
    Args:
        filepath: 檔案路徑
    
    Returns:
        JSON 物件列表
    
    Example:
        >>> cases = load_jsonl("eval_cases.jsonl")
        >>> print(len(cases))
        5
    """
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def save_jsonl(data: list[dict], filepath: str) -> None:
    """
    儲存資料為 JSONL 檔案
    
    Args:
        data: JSON 物件列表
        filepath: 輸出檔案路徑
    
    Example:
        >>> outputs = [{"id": "1", "response": "hello"}]
        >>> save_jsonl(outputs, "outputs.jsonl")
    """
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
