import json
from pathlib import Path
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8299/v1",
    api_key="EMPTY",
)

MODEL_NAME = "Qwen2.5-3B-Instruct"
INPUT_FILE = "eval_cases.jsonl"
OUTPUT_FILE = "baseline_outputs.jsonl"


def load_jsonl(path: str):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_messages(case):
    system_prompt = (
        "你是一個工具呼叫助手。"
        "請根據使用者需求，輸出一個 JSON 物件，不要輸出其他文字。"
        '格式必須為：{"tool": "...", "arguments": {...}}'
    )
    return [
        {"role": "system", "content": system_prompt},
        # TODO: 把 case["messages"] 接上來
        *case["messages"],
    ]


def run_inference(case):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=build_messages(case),
        temperature=0,
        max_tokens=128,
    )
    # TODO: 回傳模型輸出的文字
    return response.choices[0].message.content


def main():
    if not Path(INPUT_FILE).exists():
        raise FileNotFoundError(f"找不到輸入檔案: {INPUT_FILE}")

    cases = load_jsonl(INPUT_FILE)
    outputs = []

    for case in cases:
        print(f"Running case: {case['id']}")
        try:
            prediction = run_inference(case)
        except Exception as e:
            print(f"[ERROR] {case['id']}: {e}")
            prediction = ""

        outputs.append({
            "id": case["id"],
            "messages": case["messages"],
            "predict": prediction,
            "expect": case["expect"],
        })

    save_jsonl(OUTPUT_FILE, outputs)
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()