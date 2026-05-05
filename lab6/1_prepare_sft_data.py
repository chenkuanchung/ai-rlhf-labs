import json
import re
import os
from datasets import load_dataset

# 設定輸出檔案路徑
OUTPUT_FILE = "sft_data.json"
# 取訓練集的前 200 題作為 SFT 冷起動資料
NUM_SAMPLES = 200 

def clean_reasoning(text: str) -> str:
    """
    清除 GSM8K 解答中的計算機標籤，例如把 '<<16-3-4=9>>' 移除
    讓推論過程看起來更自然。
    """
    return re.sub(r'<<.*?>>', '', text).strip()

def prepare_sft_data():
    print(f"📥 正在下載/載入 GSM8K 資料集 (取前 {NUM_SAMPLES} 筆 train split)...")
    # 載入 GSM8K 訓練集
    dataset = load_dataset("openai/gsm8k", "main", split=f"train[:{NUM_SAMPLES}]")
    
    sft_data = []
    
    print("⚙️ 正在轉換資料格式 (加入 <think> 標籤)...")
    for item in dataset:
        question = item["question"]
        raw_answer = item["answer"]
        
        # GSM8K 的答案格式通常是 "推理過程 #### 最終答案"
        if "####" not in raw_answer:
            continue
            
        parts = raw_answer.split("####")
        reasoning = parts[0].strip()
        final_answer = parts[1].strip()
        
        # 清除雜訊
        clean_reason = clean_reasoning(reasoning)
        
        # 組合包含 <think> 的 assistant 回覆
        assistant_content = f"<think>\n{clean_reason}\n</think>\n{final_answer}"
        
        # 建立 messages 結構
        message_dict = {
            "messages": [
                {
                    "role": "system", 
                    "content": "你是數學助教。請先在 <think> 標籤內逐步推理，再給出最終答案。"
                },
                {
                    "role": "user", 
                    "content": question
                },
                {
                    "role": "assistant", 
                    "content": assistant_content
                }
            ]
        }
        
        sft_data.append(message_dict)

    # 儲存為 JSONL 格式（每行一個 JSON，適合 Hugging Face / TRL 訓練）
    # 這裡我們為了易讀性先存成標準的 JSON Array
    print(f"💾 正在儲存資料至 {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sft_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ 完成！成功準備了 {len(sft_data)} 筆 SFT 訓練資料。")
    
    # 印出第一筆資料作為檢查
    print("\n🔍 第一筆轉換後的資料預覽：")
    print(json.dumps(sft_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    prepare_sft_data()