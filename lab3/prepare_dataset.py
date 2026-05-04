"""
==============================================================================
Lab 3：準備 GRPO 訓練資料
==============================================================================

📌 本檔案功能：
    準備 GRPO 訓練所需的 prompt 資料集，覆蓋全部 10 個工具：
    1. get_order_status         查詢訂單狀態
    2. track_shipment           查詢物流
    3. create_refund_request    建立退款申請
    4. cancel_order             取消訂單
    5. get_order_items          查詢訂單商品明細
    6. update_shipping_address  修改配送地址
    7. get_refund_status        查詢退款案件進度
    8. apply_coupon             套用優惠碼
    9. check_product_stock      查詢商品庫存
    10. escalate_to_human       轉接真人客服

    GRPO 訓練只需要 prompts，不需要預先準備回答。
    訓練過程中，模型會自己生成多個回答，再用 reward function 評分。

📖 資料格式：
    [
        {"prompt": "...", "task_type": "...", "expected_tool": "...", "metadata": {...}},
        ...
    ]

🔧 使用方式：
    cd lab3
    python prepare_dataset.py
"""

import sys
import os
import json
import random
import string

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import save_json
from common.prompts import system_prompt


# ==============================================================================
# 訓練 Prompt 模板
# ==============================================================================

# --- 1. 查詢訂單狀態 ---
ORDER_PROMPTS = [
    "幫我查訂單 {order_id} 目前狀態",
    "我的訂單 {order_id} 到哪了",
    "查一下 {order_id} 這筆訂單",
    "訂單編號 {order_id}，幫我看一下狀態",
    "請問訂單 {order_id} 現在是什麼情況",
    "{order_id} 這個訂單出貨了嗎",
    "我想知道 {order_id} 的訂單進度",
    "幫我確認一下訂單 {order_id}",
]

# --- 2. 查詢物流 ---
TRACKING_PROMPTS = [
    "我要查物流 {tracking_no} 到哪了",
    "物流單號 {tracking_no}，幫我追蹤一下",
    "查一下 {tracking_no} 的配送進度",
    "{tracking_no} 這個包裹到哪了",
    "請問物流 {tracking_no} 目前在哪個節點",
    "幫我看看 {tracking_no} 的物流狀態",
]

# --- 3. 退款申請（資訊不完整，應追問）---
REFUND_PROMPTS_INCOMPLETE = [
    "我要退款",
    "可以幫我辦退款嗎",
    "我想要退貨",
    "這個商品我不要了",
    "我想申請退款",
]

# --- 3. 退款申請（資訊完整）---
REFUND_PROMPTS_COMPLETE = [
    "我要退款，訂單 {order_id}，原因是{reason}",
    "訂單 {order_id} 我要申請退款，因為{reason}",
    "幫我處理退款，訂單編號 {order_id}，{reason}",
    "{order_id} 這筆我想退，理由：{reason}",
]
REFUND_REASONS = ["商品瑕疵", "未收到貨", "買錯/不需要", "重複下單"]

# --- 4. 取消訂單 ---
CANCEL_ORDER_PROMPTS = [
    "我要取消訂單 {order_id}",
    "幫我把 {order_id} 這筆訂單取消",
    "訂單 {order_id} 我不買了，請幫我取消",
    "請取消訂單編號 {order_id}",
    "{order_id} 這個訂單我想取消掉",
]

# --- 5. 查詢訂單商品明細 ---
ORDER_ITEMS_PROMPTS = [
    "訂單 {order_id} 我買了哪些東西",
    "幫我列一下 {order_id} 的商品明細",
    "{order_id} 這筆訂單的內容是什麼",
    "我想看訂單 {order_id} 買了什麼、總共多少錢",
    "查一下訂單 {order_id} 的品項清單",
    "訂單編號 {order_id}，幫我看品項和金額",
]

# --- 6. 修改配送地址（完整資訊）---
UPDATE_ADDRESS_PROMPTS_COMPLETE = [
    "幫我把訂單 {order_id} 的地址改成：收件人 {recipient}，電話 {phone}，地址 {address}",
    "訂單 {order_id} 改地址，{recipient}，{phone}，{address}",
    "請更新訂單 {order_id} 的配送資訊：收件人{recipient}、電話{phone}、地址{address}",
    "{order_id} 改寄到 {address}，收件人 {recipient}，電話 {phone}",
]

# --- 6. 修改配送地址（資訊不完整，應追問）---
UPDATE_ADDRESS_PROMPTS_INCOMPLETE = [
    "我要改地址",
    "可以幫我換配送地址嗎",
    "訂單 {order_id} 我想改地址",
    "幫我改一下 {order_id} 的收件人",
    "我換了新地址，幫我更新一下",
]

# 假名與假地址
FAKE_RECIPIENTS = ["王小明", "李大華", "陳怡君", "張志偉", "林佳玲", "黃彥宏", "吳秀英", "蔡明哲"]
FAKE_ADDRESSES = [
    "台北市信義區松仁路100號5樓",
    "新北市板橋區文化路二段50號",
    "桃園市中壢區中央路200號8樓之2",
    "台中市西屯區台灣大道三段99號",
    "台南市東區東門路一段123號",
    "高雄市苓雅區四維三路6號12樓",
    "新竹市東區光復路二段88號",
]

# --- 7. 查詢退款案件進度 ---
REFUND_STATUS_PROMPTS = [
    "退款案件 {case_id} 進度怎麼樣了",
    "幫我查退款 {case_id} 的處理狀態",
    "{case_id} 這個退款案件審核好了嗎",
    "請問退款編號 {case_id} 目前進度",
    "我想知道 {case_id} 退款處理到哪了",
    "查一下退款案件 {case_id}",
]

# --- 8. 套用優惠碼 ---
APPLY_COUPON_PROMPTS = [
    "幫我把優惠碼 {coupon_code} 套用到訂單 {order_id}",
    "訂單 {order_id} 用 {coupon_code} 這個優惠碼",
    "我有一張優惠券 {coupon_code}，幫我套用到 {order_id}",
    "{order_id} 套折扣碼 {coupon_code}",
    "請幫訂單 {order_id} 使用優惠碼 {coupon_code}",
]

# --- 9. 查詢商品庫存 ---
CHECK_STOCK_PROMPTS = [
    "商品 {sku} 還有貨嗎",
    "幫我查 {sku} 的庫存",
    "{sku} 這個商品目前有多少存量",
    "請問 {sku} 缺貨了嗎，什麼時候補貨",
    "{sku} 還有沒有現貨",
    "查一下 SKU {sku} 的庫存狀況",
]

# --- 10. 轉接真人客服 ---
ESCALATE_PROMPTS = [
    ("帳務", "我發票開錯了，要重新開立"),
    ("帳務", "刷卡被多扣款，幫我處理"),
    ("物流", "物流司機態度很差，我要投訴"),
    ("物流", "包裹寄到錯的地址，已經反映多次都沒解決"),
    ("商品", "商品說明跟實際不符，我要申訴"),
    ("商品", "收到的商品有安全疑慮"),
    ("退換貨", "退款申請被駁回，我不能接受，要申訴"),
    ("退換貨", "退貨流程複雜，我想直接跟客服談"),
    ("其他", "我要對你們公司提出客訴"),
    ("其他", "這個問題你解決不了，幫我轉真人"),
]


# ==============================================================================
# 隨機資料生成器
# ==============================================================================

def generate_order_id() -> str:
    """生成隨機訂單編號（A + 9 位數字）"""
    return f"A{random.randint(100000000, 999999999)}"


def generate_tracking_no() -> str:
    """生成隨機物流單號（TWD + 8 位數字）"""
    return f"TWD{random.randint(10000000, 99999999)}"


def generate_case_id() -> str:
    """生成隨機退款案件編號（R + 6 位數字）"""
    return f"R{random.randint(100000, 999999)}"


def generate_sku() -> str:
    """生成隨機商品編號（SKU + 6 位數字）"""
    return f"SKU{random.randint(0, 999999):06d}"


def generate_phone() -> str:
    """生成隨機台灣手機號碼（09 + 8 位數字）"""
    return f"09{random.randint(10000000, 99999999)}"


def generate_coupon_code() -> str:
    """生成隨機優惠碼（4~20 碼大寫英數）"""
    presets = ["WELCOME100", "VIP500", "SUMMER2026", "NEWYEAR888",
               "FREESHIP", "SAVE20", "BIGSALE", "MEMBER10"]
    if random.random() < 0.7:
        return random.choice(presets)
    length = random.randint(6, 12)
    chars = string.ascii_uppercase + string.digits
    return "".join(random.choices(chars, k=length))


# ==============================================================================
# 各工具的 prompt 生成器
# ==============================================================================

def _make(prompt_text: str, sys_prompt: str, task_type: str,
          expected_tool: str | None, metadata: dict,
          should_clarify: bool = False) -> dict:
    """共用的 prompt 條目組裝器。"""
    item = {
        "prompt": format_chat_prompt(sys_prompt, prompt_text),
        "task_type": task_type,
        "expected_tool": expected_tool,
        "metadata": metadata,
    }
    if should_clarify:
        item["should_clarify"] = True
    return item


def gen_order_query(sys_prompt: str) -> dict:
    order_id = generate_order_id()
    text = random.choice(ORDER_PROMPTS).format(order_id=order_id)
    return _make(text, sys_prompt, "order_query", "get_order_status",
                 {"order_id": order_id})


def gen_tracking_query(sys_prompt: str) -> dict:
    tracking_no = generate_tracking_no()
    text = random.choice(TRACKING_PROMPTS).format(tracking_no=tracking_no)
    return _make(text, sys_prompt, "tracking_query", "track_shipment",
                 {"tracking_no": tracking_no})


def gen_refund_incomplete(sys_prompt: str) -> dict:
    text = random.choice(REFUND_PROMPTS_INCOMPLETE)
    return _make(text, sys_prompt, "refund_incomplete", None, {},
                 should_clarify=True)


def gen_refund_complete(sys_prompt: str) -> dict:
    order_id = generate_order_id()
    reason = random.choice(REFUND_REASONS)
    text = random.choice(REFUND_PROMPTS_COMPLETE).format(
        order_id=order_id, reason=reason)
    return _make(text, sys_prompt, "refund_complete", "create_refund_request",
                 {"order_id": order_id, "reason": reason})


def gen_cancel_order(sys_prompt: str) -> dict:
    order_id = generate_order_id()
    text = random.choice(CANCEL_ORDER_PROMPTS).format(order_id=order_id)
    return _make(text, sys_prompt, "cancel_order", "cancel_order",
                 {"order_id": order_id})


def gen_order_items(sys_prompt: str) -> dict:
    order_id = generate_order_id()
    text = random.choice(ORDER_ITEMS_PROMPTS).format(order_id=order_id)
    return _make(text, sys_prompt, "order_items", "get_order_items",
                 {"order_id": order_id})


def gen_update_address_complete(sys_prompt: str) -> dict:
    order_id = generate_order_id()
    recipient = random.choice(FAKE_RECIPIENTS)
    phone = generate_phone()
    address = random.choice(FAKE_ADDRESSES)
    text = random.choice(UPDATE_ADDRESS_PROMPTS_COMPLETE).format(
        order_id=order_id, recipient=recipient, phone=phone, address=address)
    return _make(text, sys_prompt, "update_address_complete",
                 "update_shipping_address",
                 {"order_id": order_id, "recipient": recipient,
                  "phone": phone, "address": address})


def gen_update_address_incomplete(sys_prompt: str) -> dict:
    template = random.choice(UPDATE_ADDRESS_PROMPTS_INCOMPLETE)
    metadata = {}
    if "{order_id}" in template:
        order_id = generate_order_id()
        text = template.format(order_id=order_id)
        metadata["order_id"] = order_id
    else:
        text = template
    return _make(text, sys_prompt, "update_address_incomplete", None, metadata,
                 should_clarify=True)


def gen_refund_status(sys_prompt: str) -> dict:
    case_id = generate_case_id()
    text = random.choice(REFUND_STATUS_PROMPTS).format(case_id=case_id)
    return _make(text, sys_prompt, "refund_status", "get_refund_status",
                 {"case_id": case_id})


def gen_apply_coupon(sys_prompt: str) -> dict:
    order_id = generate_order_id()
    coupon_code = generate_coupon_code()
    text = random.choice(APPLY_COUPON_PROMPTS).format(
        order_id=order_id, coupon_code=coupon_code)
    return _make(text, sys_prompt, "apply_coupon", "apply_coupon",
                 {"order_id": order_id, "coupon_code": coupon_code})


def gen_check_stock(sys_prompt: str) -> dict:
    sku = generate_sku()
    text = random.choice(CHECK_STOCK_PROMPTS).format(sku=sku)
    return _make(text, sys_prompt, "check_stock", "check_product_stock",
                 {"sku": sku})


def gen_escalate(sys_prompt: str) -> dict:
    topic, summary = random.choice(ESCALATE_PROMPTS)
    # 一半機率附帶訂單編號
    metadata = {"topic": topic, "summary": summary}
    if random.random() < 0.5:
        order_id = generate_order_id()
        text = f"{summary}（訂單 {order_id}），請幫我轉真人客服"
        metadata["order_id"] = order_id
    else:
        text = f"{summary}，請幫我轉真人客服處理"
    return _make(text, sys_prompt, "escalate", "escalate_to_human", metadata)


# 工具類別 → 生成器對照表
TASK_GENERATORS = {
    "order_query":               gen_order_query,
    "tracking_query":            gen_tracking_query,
    "refund_incomplete":         gen_refund_incomplete,
    "refund_complete":           gen_refund_complete,
    "cancel_order":              gen_cancel_order,
    "order_items":               gen_order_items,
    "update_address_complete":   gen_update_address_complete,
    "update_address_incomplete": gen_update_address_incomplete,
    "refund_status":             gen_refund_status,
    "apply_coupon":              gen_apply_coupon,
    "check_stock":               gen_check_stock,
    "escalate":                  gen_escalate,
}


# ==============================================================================
# 主要資料集生成函式
# ==============================================================================

# 各任務預設數量（共 12 類，覆蓋 10 個工具 + 2 種需追問情境）
DEFAULT_COUNTS = {
    "order_query":               20,
    "tracking_query":            15,
    "refund_incomplete":         8,
    "refund_complete":           12,
    "cancel_order":              12,
    "order_items":               12,
    "update_address_complete":   10,
    "update_address_incomplete": 6,
    "refund_status":             10,
    "apply_coupon":              10,
    "check_stock":               12,
    "escalate":                  8,
}


def create_training_prompts(counts: dict[str, int] | None = None) -> list[dict]:
    """
    建立訓練用的 prompt 資料集

    Args:
        counts: 各任務類別的數量。若為 None 則使用 DEFAULT_COUNTS。
                key 必須是 TASK_GENERATORS 中的任務名稱。

    Returns:
        包含 prompt 的字典列表
    """
    if counts is None:
        counts = DEFAULT_COUNTS

    sys_prompt = system_prompt()
    prompts = []

    for task_type, n in counts.items():
        if task_type not in TASK_GENERATORS:
            raise ValueError(f"未知的 task_type: {task_type}")
        gen = TASK_GENERATORS[task_type]
        for _ in range(n):
            prompts.append(gen(sys_prompt))

    random.shuffle(prompts)
    return prompts


def format_chat_prompt(system_content: str, user_content: str) -> str:
    """
    將對話格式化為模型輸入格式（ChatML，Qwen 模型使用）：
    <|im_start|>system
    {system_content}<|im_end|>
    <|im_start|>user
    {user_content}<|im_end|>
    <|im_start|>assistant
    """
    return f"""<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""


def main():
    print("=" * 60)
    print("Lab 3：準備 GRPO 訓練資料（10 個工具）")
    print("=" * 60)

    # 建立訓練 prompts
    print("\n📝 生成訓練 prompts...")
    prompts = create_training_prompts()

    print(f"   總共生成 {len(prompts)} 個 prompts")

    # 統計各類型數量
    task_counts: dict[str, int] = {}
    for p in prompts:
        task_type = p.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    print("\n📊 各任務類型統計：")
    for task_type, count in sorted(task_counts.items()):
        print(f"   - {task_type:<28s}: {count}")

    # 統計各工具被覆蓋的數量
    tool_counts: dict[str, int] = {}
    for p in prompts:
        tool = p.get("expected_tool") or "(追問/不呼叫工具)"
        tool_counts[tool] = tool_counts.get(tool, 0) + 1

    print("\n🔧 各工具覆蓋統計：")
    for tool, count in sorted(tool_counts.items()):
        print(f"   - {tool:<28s}: {count}")

    # 儲存訓練資料
    output_file = "training_prompts.json"
    save_json(prompts, output_file)
    print(f"\n💾 已儲存至 {output_file}")

    # 顯示範例
    print("\n📋 範例 prompt（前 5 個）：")
    for i, p in enumerate(prompts[:5]):
        print(f"\n--- 範例 {i+1} ---")
        print(f"類型：{p['task_type']}")
        print(f"期望工具：{p.get('expected_tool') or '追問'}")
        user_start = p['prompt'].find("<|im_start|>user\n") + len("<|im_start|>user\n")
        user_end = p['prompt'].find("<|im_end|>\n<|im_start|>assistant")
        user_content = p['prompt'][user_start:user_end]
        print(f"使用者輸入：{user_content}")

    print("\n✅ 訓練資料準備完成！")
    print("   下一步：前往 lab4 執行 GRPO 訓練")


if __name__ == "__main__":
    main()