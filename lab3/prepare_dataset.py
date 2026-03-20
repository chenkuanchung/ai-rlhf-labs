"""
==============================================================================
Lab 3：準備 GRPO 訓練資料
==============================================================================

📌 本檔案功能：
    準備 GRPO 訓練所需的 prompt 資料集。
    
    GRPO 訓練只需要 prompts，不需要預先準備回答。
    訓練過程中，模型會自己生成多個回答，再用 reward function 評分。

📖 資料格式：
    GRPO 訓練資料非常簡單，只需要 prompt：
    [
        {"prompt": "幫我查訂單 A123456789 目前狀態"},
        {"prompt": "我要查物流 TWD12345678 到哪了"},
        ...
    ]

🔧 使用方式：
    cd lab3
    python 1_prepare_dataset.py
"""

import sys
import os
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.utils import save_jsonl, load_jsonl
from common.prompts import system_prompt


# ==============================================================================
# 訓練 Prompt 模板
# ==============================================================================

# 查詢訂單狀態的 prompt 模板
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

# 查詢物流的 prompt 模板
TRACKING_PROMPTS = [
    "我要查物流 {tracking_no} 到哪了",
    "物流單號 {tracking_no}，幫我追蹤一下",
    "查一下 {tracking_no} 的配送進度",
    "{tracking_no} 這個包裹到哪了",
    "請問物流 {tracking_no} 目前在哪個節點",
    "幫我看看 {tracking_no} 的物流狀態",
]

# 退款相關的 prompt 模板（需要追問的情況）
REFUND_PROMPTS_INCOMPLETE = [
    "我要退款",
    "可以幫我辦退款嗎",
    "我想要退貨",
    "這個商品我不要了",
]

# 退款相關的 prompt 模板（資訊完整）
REFUND_PROMPTS_COMPLETE = [
    "我要退款，訂單 {order_id}，原因是{reason}",
    "訂單 {order_id} 我要申請退款，因為{reason}",
    "幫我處理退款，訂單編號 {order_id}，{reason}",
]

REFUND_REASONS = ["商品瑕疵", "未收到貨", "買錯/不需要", "重複下單"]


def generate_order_id() -> str:
    """生成隨機訂單編號"""
    return f"A{random.randint(100000000, 999999999)}"


def generate_tracking_no() -> str:
    """生成隨機物流單號"""
    return f"TWD{random.randint(10000000, 99999999)}"


def create_training_prompts(
    num_order: int = 30,
    num_tracking: int = 20,
    num_refund_incomplete: int = 10,
    num_refund_complete: int = 15
) -> list[dict]:
    """
    建立訓練用的 prompt 資料集
    
    Args:
        num_order: 查詢訂單的 prompt 數量
        num_tracking: 查詢物流的 prompt 數量
        num_refund_incomplete: 需要追問的退款 prompt 數量
        num_refund_complete: 完整退款 prompt 數量
    
    Returns:
        包含 prompt 的字典列表
    """
    
    prompts = []
    sys_prompt = system_prompt()
    
    # 生成查詢訂單的 prompts
    for _ in range(num_order):
        template = random.choice(ORDER_PROMPTS)
        order_id = generate_order_id()
        user_content = template.format(order_id=order_id)
        
        # GRPO 需要完整的對話格式
        prompts.append({
            "prompt": format_chat_prompt(sys_prompt, user_content),
            "task_type": "order_query",
            "expected_tool": "get_order_status",
            "metadata": {"order_id": order_id}
        })
    
    # 生成查詢物流的 prompts
    for _ in range(num_tracking):
        template = random.choice(TRACKING_PROMPTS)
        tracking_no = generate_tracking_no()
        user_content = template.format(tracking_no=tracking_no)
        
        prompts.append({
            "prompt": format_chat_prompt(sys_prompt, user_content),
            "task_type": "tracking_query",
            "expected_tool": "track_shipment",
            "metadata": {"tracking_no": tracking_no}
        })
    
    # 生成需要追問的退款 prompts
    for _ in range(num_refund_incomplete):
        user_content = random.choice(REFUND_PROMPTS_INCOMPLETE)
        
        prompts.append({
            "prompt": format_chat_prompt(sys_prompt, user_content),
            "task_type": "refund_incomplete",
            "expected_tool": None,  # 應該追問，不應該呼叫工具
            "should_clarify": True,
            "metadata": {}
        })
    
    # 生成完整的退款 prompts
    for _ in range(num_refund_complete):
        template = random.choice(REFUND_PROMPTS_COMPLETE)
        order_id = generate_order_id()
        reason = random.choice(REFUND_REASONS)
        user_content = template.format(order_id=order_id, reason=reason)
        
        prompts.append({
            "prompt": format_chat_prompt(sys_prompt, user_content),
            "task_type": "refund_complete",
            "expected_tool": "create_refund_request",
            "metadata": {"order_id": order_id, "reason": reason}
        })
    
    # 打亂順序
    random.shuffle(prompts)
    
    return prompts


def format_chat_prompt(system_content: str, user_content: str) -> str:
    """
    將對話格式化為模型輸入格式
    
    使用 ChatML 格式，這是 Qwen 模型使用的格式：
    <|im_start|>system
    {system_content}<|im_end|>
    <|im_start|>user
    {user_content}<|im_end|>
    <|im_start|>assistant
    
    Args:
        system_content: system prompt 內容
        user_content: 使用者訊息內容
    
    Returns:
        格式化的 prompt 字串
    """
    return f"""<|im_start|>system
{system_content}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""


def main():
    """
    主程式
    """
    print("=" * 60)
    print("Lab 3：準備 GRPO 訓練資料")
    print("=" * 60)
    
    # 建立訓練 prompts
    print("\n📝 生成訓練 prompts...")
    prompts = create_training_prompts(
        num_order=30,
        num_tracking=20,
        num_refund_incomplete=10,
        num_refund_complete=15
    )
    
    print(f"   總共生成 {len(prompts)} 個 prompts")
    
    # 統計各類型數量
    task_counts = {}
    for p in prompts:
        task_type = p.get("task_type", "unknown")
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    print("\n📊 各類型統計：")
    for task_type, count in sorted(task_counts.items()):
        print(f"   - {task_type}: {count}")
    
    # 儲存訓練資料
    output_file = "training_prompts.jsonl"
    save_jsonl(prompts, output_file)
    print(f"\n💾 已儲存至 {output_file}")
    
    # 顯示範例
    print("\n📋 範例 prompt（前 3 個）：")
    for i, p in enumerate(prompts[:3]):
        print(f"\n--- 範例 {i+1} ---")
        print(f"類型：{p['task_type']}")
        print(f"期望工具：{p.get('expected_tool', '追問')}")
        # 只顯示 user 部分
        user_start = p['prompt'].find("<|im_start|>user\n") + len("<|im_start|>user\n")
        user_end = p['prompt'].find("<|im_end|>\n<|im_start|>assistant")
        user_content = p['prompt'][user_start:user_end]
        print(f"使用者輸入：{user_content}")
    
    print("\n✅ 訓練資料準備完成！")
    print("   下一步：前往 lab4 執行 GRPO 訓練")


if __name__ == "__main__":
    main()
