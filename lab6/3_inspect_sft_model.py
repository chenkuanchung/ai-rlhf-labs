import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ==========================================
# 設定區
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "./sft_output/final"

def main():
    print("=" * 50)
    print("🔍 Lab 6: 觀察 SFT 冷起動後的模型表現")
    print("=" * 50)

    # 1. 載入 Tokenizer
    print("📦 正在載入 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    # 2. 載入基礎模型
    # 推論時我們可以用 bfloat16 載入，3B 模型大約只佔用 6GB VRAM，速度比 4-bit 快
    print("📦 正在載入基礎模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 3. 掛載 SFT 訓練出的 LoRA 權重
    print(f"🔧 正在掛載 SFT LoRA 權重從：{ADAPTER_PATH}...")
    try:
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    except Exception as e:
        print(f"\n❌ 載入失敗！請確認你已經跑完 2_sft_training.py，並且 {ADAPTER_PATH} 存在。")
        print(f"錯誤訊息: {e}")
        return

    # 4. 準備測試題目
    system_prompt = "你是數學助教。請先在 <think> 標籤內逐步推理，再給出最終答案。"
    
    test_questions = [
        # 第一題：經典的 GSM8K 測試題 (Janet's ducks)
        "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
        
        # 第二題：模型沒看過的中文數學題 (測試舉一反三的能力)
        "小明有 150 元，買了 3 瓶 25 元的果汁，剩下的錢可以買幾個 15 元的麵包？"
    ]

    print("\n🚀 開始進行推論測試...\n")

    for idx, question in enumerate(test_questions, 1):
        # 組合訊息
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ]
        
        # 套用 Chat Template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)

        print(f"[{idx}/2] ❓ 題目: {question}")
        print("⏳ 模型思考中 (這可能需要幾秒鐘)...")
        
        # 產生回答
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,  # 給予足夠長度輸出 think 和答案
                temperature=0.7,     # 稍微給一點隨機性
                do_sample=True,
                top_p=0.9
            )
        
        # 解碼輸出結果 (排除掉輸入的 prompt 部分)
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print("\n💬 模型回答:")
        print("-" * 40)
        print(response)
        print("-" * 40 + "\n")

if __name__ == "__main__":
    main()