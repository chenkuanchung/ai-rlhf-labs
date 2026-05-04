import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from common.call_llm import call_llm
from common.prompts import system_prompt

LAB_DIR = Path(__file__).parent
INPUT_FILE = LAB_DIR / "eval_cases.json"
OUTPUT_FILE = LAB_DIR / "baseline_outputs.json"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def run_inference(case):
    messages = [
        # TODO: 複習:如同Course7，  先加入 system prompt 



        # TODO: 把 case["messages"] 接上來
    ]
    return call_llm(messages)


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
        
        print(prediction)
        # TODO: 觀察prediction裡面是否有reasoning回覆 </think>
        # 請移除reasoning的部分，留下tool call回復或是模型的最終回答

        
        outputs.append({
            "id": case["id"],
            "messages": case["messages"],
            "predict": prediction,
            "expect": case["expect"],
        })

    save_json(OUTPUT_FILE, outputs)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()