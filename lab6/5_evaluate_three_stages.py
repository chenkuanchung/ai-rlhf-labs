import json
import re
import torch
import gc
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

# ==========================================
# 設定區
# ==========================================
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SFT_ADAPTER = "./sft_output/final"
GRPO_ADAPTER = "./grpo_output/final"
TEST_SAMPLES = 50  # 為了節省時間，我們先測 50 題。想看完整結果可改為 200

# ==========================================
# 評估與萃取邏輯
# ==========================================
def extract_ground_truth(answer_str: str) -> str:
    """從 GSM8K 的原始 answer 中抽出最終數字"""
    return answer_str.split("####")[1].strip()

def evaluate_response(response: str, gt_answer: str):
    """評估單一回答的格式與正確率"""
    has_think_tags = False
    think_length = 0
    is_correct = False
    
    # 1. 檢查格式與萃取 think 長度
    think_match = re.search(r'<think>(.*?)</think>', response, flags=re.DOTALL)
    if think_match:
        has_think_tags = True
        think_length = len(think_match.group(1).strip())
        # 取出 </think> 之後的文字作為最終答案區塊
        tail_text = response.split("</think>")[-1]
    else:
        tail_text = response
        
    # 2. 檢查正確率 (取出 tail_text 中的最後一個數字)
    nums = re.findall(r'-?\d+(?:\.\d+)?', tail_text.replace(',', ''))
    if nums and nums[-1] == gt_answer:
        is_correct = True
        
    return has_think_tags, think_length, is_correct

# ==========================================
# 核心執行函數
# ==========================================
def run_evaluation(model_name_or_path, adapter_path=None, stage_name="Unknown"):
    print(f"\n[{stage_name}] 📥 正在載入模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # 載入基礎模型 (使用 bfloat16 以加快推論)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # 如果有 adapter 則掛載
    if adapter_path:
        print(f"[{stage_name}] 🔧 掛載 Adapter: {adapter_path}")
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model
        
    model.eval()
    
    print(f"[{stage_name}] 📥 準備測試集 ({TEST_SAMPLES} 題)...")
    dataset = load_dataset("openai/gsm8k", "main", split=f"test[:{TEST_SAMPLES}]")
    
    system_prompt = "你是數學助教。請先在 <think> 標籤內逐步推理，再給出最終答案。"
    
    results = {
        "format_correct": 0,
        "correct_answers": 0,
        "total_think_length": 0,
        "total": len(dataset)
    }
    
    print(f"[{stage_name}] 🚀 開始評測...")
    for item in tqdm(dataset):
        gt = extract_ground_truth(item["answer"])
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": item["question"]}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            # 使用 Greedy Decoding (temperature=0.0) 測試最穩定的表現
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False 
            )
            
        generated_ids = outputs[0][len(inputs.input_ids[0]):]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # 計算分數
        has_tags, think_len, is_correct = evaluate_response(response, gt)
        if has_tags: results["format_correct"] += 1
        if is_correct: results["correct_answers"] += 1
        results["total_think_length"] += think_len

    # 釋放記憶體，這是跑下一個模型的關鍵！
    print(f"[{stage_name}] 🧹 卸載模型與清理 VRAM...")
    del model
    del base_model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

# ==========================================
# 主程式
# ==========================================
def main():
    print("=" * 60)
    print("🏆 Lab 6: 三階段推理能力終極評測 (Base vs SFT vs GRPO)")
    print("=" * 60)
    
    reports = {}
    
    # 測試 Stage 0: Base 模型 (未經任何格式訓練)
    try:
        reports["Base"] = run_evaluation(MODEL_NAME, adapter_path=None, stage_name="Stage 0: Base")
    except Exception as e:
        print(f"Base 模型評測失敗: {e}")
        
    # 測試 Stage 1: SFT 模型 (學會格式，但未經 RL 強化)
    try:
        reports["SFT"] = run_evaluation(MODEL_NAME, adapter_path=SFT_ADAPTER, stage_name="Stage 1: SFT")
    except Exception as e:
        print(f"SFT 模型評測失敗 (請確認 SFT 訓練已完成): {e}")
        
    # 測試 Stage 2: GRPO 模型 (SFT 加上強化學習)
    try:
        reports["GRPO"] = run_evaluation(MODEL_NAME, adapter_path=GRPO_ADAPTER, stage_name="Stage 2: GRPO")
    except Exception as e:
        print(f"GRPO 模型評測失敗 (請確認 GRPO 訓練已完成): {e}")

    # ==========================================
    # 產出漂亮的比對表格
    # ==========================================
    print("\n" + "=" * 60)
    print(f"{'模型階段':<15} | {'Format Rate':<12} | {'Accuracy':<10} | {'Avg Think Len':<15}")
    print("-" * 60)
    
    for stage in ["Base", "SFT", "GRPO"]:
        if stage in reports:
            res = reports[stage]
            total = res["total"]
            fmt_rate = (res["format_correct"] / total) * 100
            acc = (res["correct_answers"] / total) * 100
            # 只有當有輸出的時候才算平均長度
            avg_len = res["total_think_length"] / res["format_correct"] if res["format_correct"] > 0 else 0
            
            print(f"{stage:<15} | {fmt_rate:>5.1f}%       | {acc:>5.1f}%     | {avg_len:>8.0f} 字")

    print("=" * 60)
    print("💡 觀察指南：")
    print("1. Format Rate: SFT 階段應該要從 0% 暴增到接近 100%。")
    print("2. Accuracy: GRPO 階段的準確率應該要超越 SFT。")
    print("3. Avg Think Len: 通常 GRPO 後，模型會『想得更深』，因此平均字數會變多。")

if __name__ == "__main__":
    main()