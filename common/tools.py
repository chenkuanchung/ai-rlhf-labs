"""
==============================================================================
Common 模組：Mock 工具實作 (tools.py)
==============================================================================

📌 本檔案功能：
    提供模擬的工具函式（Mock Tools），用於開發和測試。
    
    在實際部署中，這些函式會連接到真正的後端服務（資料庫、API 等）。
    但在 Lab 中，我們用 Mock 資料來模擬行為。

📖 Mock 資料的價值：
    1. 不需要真實後端也能開發和測試
    2. 可以控制各種情境（成功、失敗、找不到等）
    3. 快速迭代，不用擔心外部依賴

🔧 使用方式：
    from common.tools import TOOL_REGISTRY
    
    # 透過名稱呼叫工具
    result = TOOL_REGISTRY["get_order_status"](order_id="A123456789")
    print(result)
    # {"ok": True, "order_id": "A123456789", "status": "已出貨", ...}

📋 新增工具步驟：
    1. 在本檔案定義函式
    2. 在 TOOL_REGISTRY 中註冊
    3. 在 tool_schema.py 中定義 Schema
"""

from typing import Dict, Any
import random
import time


# ==============================================================================
# Mock 資料庫
# ==============================================================================

ORDERS = {
    "A123456789": {"status": "已出貨", "tracking_no": "TWD12345678"},
    "A000000001": {"status": "處理中", "tracking_no": None},
    "A999999999": {"status": "已取消", "tracking_no": None},
}
"""
模擬的訂單資料庫

格式：
    {
        "訂單編號": {
            "status": "訂單狀態",
            "tracking_no": "物流單號（可能為 None）"
        }
    }

測試案例：
    - A123456789：已出貨，有物流單號
    - A000000001：處理中，尚無物流單號
    - A999999999：已取消
    - 其他訂單編號：會回傳 ORDER_NOT_FOUND
"""

SHIPMENTS = {
    "TWD12345678": [
        {"ts": "2026-01-20 10:00", "node": "已收件"},
        {"ts": "2026-01-21 08:00", "node": "轉運中心"},
        {"ts": "2026-01-22 15:20", "node": "配送中"},
    ]
}
"""
模擬的物流資料庫

格式：
    {
        "物流單號": [
            {"ts": "時間戳", "node": "物流節點"},
            ...
        ]
    }

測試案例：
    - TWD12345678：有 3 個物流節點
    - 其他物流單號：會回傳 TRACKING_NOT_FOUND
"""


# ==============================================================================
# 工具函式
# ==============================================================================

def get_order_status(order_id: str) -> Dict[str, Any]:
    """
    查詢訂單狀態
    
    根據訂單編號查詢訂單的當前狀態和物流單號。
    
    Args:
        order_id: 訂單編號（格式：A + 9位數字）
    
    Returns:
        成功時：
        {
            "ok": True,
            "order_id": "A123456789",
            "status": "已出貨",
            "tracking_no": "TWD12345678"  # 可能為 None
        }
        
        失敗時：
        {
            "ok": False,
            "error": "ORDER_NOT_FOUND",
            "order_id": "A123456789"
        }
    
    Example:
        >>> get_order_status("A123456789")
        {"ok": True, "order_id": "A123456789", "status": "已出貨", "tracking_no": "TWD12345678"}
        
        >>> get_order_status("A000000000")
        {"ok": False, "error": "ORDER_NOT_FOUND", "order_id": "A000000000"}
    
    使用場景：
        - 使用者詢問「訂單 A123456789 到哪了」
        - LLM 呼叫此工具取得狀態
        - 根據狀態回覆使用者
    """
    # 模擬網路延遲（0.1 秒）
    # 在實際系統中，這是真實的 API 呼叫延遲
    time.sleep(0.1)
    
    # 查詢訂單
    if order_id not in ORDERS:
        # 找不到訂單
        return {
            "ok": False,
            "error": "ORDER_NOT_FOUND",
            "order_id": order_id
        }
    
    # 取得訂單資料
    data = ORDERS[order_id]
    
    return {
        "ok": True,
        "order_id": order_id,
        **data  # 展開 status, tracking_no
    }


def track_shipment(tracking_no: str) -> Dict[str, Any]:
    """
    查詢物流狀態
    
    根據物流單號查詢最新的配送進度。
    
    Args:
        tracking_no: 物流單號（格式：TWD + 8位數字）
    
    Returns:
        成功時：
        {
            "ok": True,
            "tracking_no": "TWD12345678",
            "events": [
                {"ts": "2026-01-20 10:00", "node": "已收件"},
                {"ts": "2026-01-21 08:00", "node": "轉運中心"},
                {"ts": "2026-01-22 15:20", "node": "配送中"}
            ]
        }
        
        失敗時：
        {
            "ok": False,
            "error": "TRACKING_NOT_FOUND",
            "tracking_no": "TWD12345678"
        }
    
    Example:
        >>> track_shipment("TWD12345678")
        {"ok": True, "tracking_no": "TWD12345678", "events": [...]}
    
    使用場景：
        - 使用者詢問「物流 TWD12345678 到哪了」
        - LLM 呼叫此工具取得物流節點
        - 根據最新節點回覆使用者
    """
    # 模擬網路延遲
    time.sleep(0.1)
    
    # 查詢物流
    if tracking_no not in SHIPMENTS:
        return {
            "ok": False,
            "error": "TRACKING_NOT_FOUND",
            "tracking_no": tracking_no
        }
    
    # 取得物流事件（最多回傳最近 3 筆）
    events = SHIPMENTS[tracking_no][-3:]
    
    return {
        "ok": True,
        "tracking_no": tracking_no,
        "events": events
    }


def create_refund_request(
    order_id: str, 
    reason: str, 
    details: str = ""
) -> Dict[str, Any]:
    """
    建立退款申請
    
    為指定訂單建立退款申請，會產生一個案件編號。
    
    Args:
        order_id: 訂單編號
        reason: 退款原因（必須是預定義的選項之一）
        details: 補充說明（可選）
    
    Returns:
        成功時：
        {
            "ok": True,
            "case_id": "R123456",
            "order_id": "A123456789",
            "reason": "商品瑕疵",
            "details": "...",
            "message": "退款申請已建立，客服將於 1-2 個工作天內處理。"
        }
        
        失敗時（訂單不存在）：
        {
            "ok": False,
            "error": "ORDER_NOT_FOUND",
            "order_id": "A123456789"
        }
        
        失敗時（訂單已取消）：
        {
            "ok": False,
            "error": "ORDER_ALREADY_CANCELLED",
            "order_id": "A999999999"
        }
    
    Example:
        >>> create_refund_request("A123456789", "商品瑕疵", "收到時螢幕有裂痕")
        {"ok": True, "case_id": "R123456", ...}
    
    使用場景：
        - 使用者說「我要退款，訂單 A123456789，因為商品壞了」
        - LLM 呼叫此工具建立退款申請
        - 回覆使用者案件編號和後續處理時間
    """
    # 模擬網路延遲
    time.sleep(0.1)
    
    # 檢查訂單是否存在
    if order_id not in ORDERS:
        return {
            "ok": False,
            "error": "ORDER_NOT_FOUND",
            "order_id": order_id
        }
    
    # 檢查訂單是否已取消
    if ORDERS[order_id]["status"] == "已取消":
        return {
            "ok": False,
            "error": "ORDER_ALREADY_CANCELLED",
            "order_id": order_id
        }
    
    # 產生退款案件編號（隨機 6 位數）
    case_id = f"R{random.randint(100000, 999999)}"
    
    return {
        "ok": True,
        "case_id": case_id,
        "order_id": order_id,
        "reason": reason,
        "details": details,
        "message": "退款申請已建立，客服將於 1-2 個工作天內處理。"
    }


# ==============================================================================
# 工具註冊表
# ==============================================================================

TOOL_REGISTRY = {
    "get_order_status": get_order_status,
    "track_shipment": track_shipment,
    "create_refund_request": create_refund_request,
}
"""
工具註冊表

將工具名稱映射到實際的函式。

使用方式：
    tool_fn = TOOL_REGISTRY["get_order_status"]
    result = tool_fn(order_id="A123456789")
    
    # 或使用動態名稱
    name = tool_call["name"]
    args = tool_call["arguments"]
    result = TOOL_REGISTRY[name](**args)

新增工具：
    1. 定義函式
    2. 在這裡註冊：TOOL_REGISTRY["new_tool"] = new_tool_fn
    3. 在 tool_schema.py 新增 Schema
"""
