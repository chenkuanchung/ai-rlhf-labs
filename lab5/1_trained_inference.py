import json
import sys
from pathlib import Path
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.call_llm import call_llm
from common.prompts import system_prompt

LAB_DIR = Path(__file__).parent
INPUT_FILE = LAB_DIR / "eval_cases.json"
OUTPUT_FILE = LAB_DIR / "trained_outputs.json"
MODEL_NAME = "qwen-grpo"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def run_inference(case):
    # TODO: 複習:如同Course7，先加入 system prompt
    # 從 common.prompts 匯入的 system_prompt 應該是一段字串，我們把它包裝成 system role
    messages = [
        {"role": "system", "content": system_prompt()}
    ]

    # TODO: 把 case["messages"] 接上來
    # 假設 case["messages"] 是一個 list of dicts (例如: [{"role": "user", "content": "..."}])
    messages.extend(case["messages"])
    
    return call_llm(messages, model=MODEL_NAME)


def main():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"找不到輸入檔案: {INPUT_FILE}")

    cases = load_json(INPUT_FILE)
    outputs = []

    for case in cases:
        print(f"Running case: {case['id']}")
        try:
            prediction = run_inference(case)
        except Exception as e:
            print(f"[ERROR] {case['id']}: {e}")
            prediction = ""
        
        print("\n--- 原始預測結果 ---")
        print(prediction)
        print("--------------------\n")

        # TODO: 觀察 prediction 裡面是否有 reasoning 回覆 </think>
        # 請移除 reasoning 的部分，留下 tool call 回復或是模型的最終回答
        
        # 使用正則表達式 (Regex) 移除 <think>...</think> 區塊
        # re.DOTALL 允許 . 匹配包括換行符號在內的所有字元
        # .*? 是非貪婪匹配，確保只匹配到最近的 </think>
        clean_prediction = re.sub(r'<think>.*?</think>', '', prediction, flags=re.DOTALL)
        
        # 移除可能殘留的開頭或結尾空白字元
        clean_prediction = clean_prediction.strip()

        outputs.append({
            "id": case["id"],
            "messages": case["messages"],
            "predict": clean_prediction, # 儲存過濾後的乾淨結果
            "expect": case["expect"],
        })

    save_json(OUTPUT_FILE, outputs)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()