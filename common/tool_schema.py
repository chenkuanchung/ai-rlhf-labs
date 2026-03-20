"""
==============================================================================
Common 模組：工具 Schema 定義 (tool_schema.py)
==============================================================================

📌 本檔案功能：
    定義所有可用工具的 JSON Schema，包括：
    1. 工具名稱和描述
    2. 參數定義（類型、格式、是否必填）
    3. 參數約束（enum、pattern 等）

📖 JSON Schema 簡介：
    JSON Schema 是描述 JSON 資料格式的標準。
    我們用它來：
    1. 告訴 LLM 每個工具需要什麼參數
    2. 驗證 LLM 輸出的參數是否正確

🔧 使用方式：
    from common.tool_schema import TOOLS
    
    # 取得所有工具定義
    for tool in TOOLS:
        print(tool["name"], tool["description"])

📋 Schema 格式說明：
    {
        "name": "tool_name",           # 工具唯一名稱
        "description": "...",          # 功能描述（LLM 靠此選工具）
        "parameters": {                # 參數 Schema
            "type": "object",
            "properties": {
                "param1": {
                    "type": "string",
                    "description": "參數描述",
                    "pattern": "正則表達式",
                    "enum": ["選項1", "選項2"]
                }
            },
            "required": ["param1"],    # 必填參數
            "additionalProperties": false  # 不允許額外參數
        }
    }

📖 各欄位說明：
    - name: 工具唯一名稱，LLM 呼叫時使用
    - description: 功能描述，這是 LLM 選擇工具的主要依據
    - parameters: 參數的 JSON Schema
      - type: 參數類型（string, number, boolean, object, array）
      - description: 參數說明，提供範例幫助 LLM 填入正確值
      - pattern: 正則表達式，用於驗證格式
      - enum: 限制參數只能是列表中的值
      - required: 必填參數列表
      - additionalProperties: 是否允許額外參數
"""


# ==============================================================================
# 工具定義
# ==============================================================================

TOOLS = [
    # --------------------------------------------------------------------------
    # 工具 1：查詢訂單狀態
    # --------------------------------------------------------------------------
    # description 是 LLM 選擇工具的主要依據，應該清楚說明工具的用途
    # 括號中列出可能的結果，幫助 LLM 理解
    {
        "name": "get_order_status",
        "description": "查詢訂單狀態（出貨/配送/已取消等）。",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    # description 提供範例格式，幫助 LLM 填入正確的值
                    # 如果使用者輸入「我的訂單是 A123456789」，
                    # LLM 應該能從中提取出訂單編號
                    "description": "訂單編號，例如 A123456789",
                    # pattern 是正則表達式，用於驗證參數格式：
                    # ^[A-Z] = 開頭是大寫字母
                    # \d{9} = 接著 9 個數字
                    # $ = 結尾
                    # 例如：A123456789 ✓, 123456789 ✗, A12345 ✗
                    "pattern": "^[A-Z]\\d{9}$"
                }
            },
            # required 列出必填的參數
            # 如果 LLM 沒有填入這些參數，驗證會失敗
            "required": ["order_id"],
            # additionalProperties: False 不允許額外的參數
            "additionalProperties": False
        },
    },
    
    # --------------------------------------------------------------------------
    # 工具 2：查詢物流狀態
    # --------------------------------------------------------------------------
    # 物流單號格式：TWD + 8 位數字，例如 TWD12345678
    {
        "name": "track_shipment",
        "description": "查詢物流最新節點（需要物流單號）。",
        "parameters": {
            "type": "object",
            "properties": {
                "tracking_no": {
                    "type": "string",
                    "description": "物流單號，例如 TWD12345678",
                    "pattern": "^TWD\\d{8}$"
                }
            },
            "required": ["tracking_no"],
            "additionalProperties": False
        },
    },
    
    # --------------------------------------------------------------------------
    # 工具 3：建立退款申請
    # --------------------------------------------------------------------------
    # 注意：order_id 和 reason 是必填，details 是選填
    # 如果使用者只說「我要退款」，LLM 應該追問 order_id 和 reason
    {
        "name": "create_refund_request",
        "description": "建立退款申請（原因與訂單編號必填）。",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "pattern": "^[A-Z]\\d{9}$"
                },
                "reason": {
                    "type": "string",
                    # enum 限制參數只能是列表中的值
                    # LLM 會從這些選項中選擇
                    # 如果使用者說「因為東西壞了」，LLM 應該對應到「商品瑕疵」
                    "enum": ["未收到貨", "商品瑕疵", "買錯/不需要", "重複下單", "其他"]
                },
                "details": {
                    "type": "string",
                    "description": "補充說明（可選）"
                },
            },
            "required": ["order_id", "reason"],
            "additionalProperties": False
        },
    },
]


# ==============================================================================
# 輔助函式
# ==============================================================================

def get_tool_by_name(name: str) -> dict | None:
    """
    根據名稱取得工具定義
    
    Args:
        name: 工具名稱
        
    Returns:
        工具定義字典，如果找不到則回傳 None
    
    Example:
        >>> tool = get_tool_by_name("get_order_status")
        >>> print(tool["description"])
        查詢訂單狀態（出貨/配送/已取消等）。
    """
    for tool in TOOLS:
        if tool["name"] == name:
            return tool
    return None


def get_tool_names() -> list[str]:
    """
    取得所有工具名稱
    
    Returns:
        工具名稱列表
    
    Example:
        >>> names = get_tool_names()
        >>> print(names)
        ['get_order_status', 'track_shipment', 'create_refund_request']
    """
    return [tool["name"] for tool in TOOLS]
