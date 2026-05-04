import json

INPUT_FILE = "eval_cases.json"


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def try_parse_json(text: str):
    try:
        return json.loads(text.strip())
    except Exception:
        return None


def evaluate(rows):
    total = len(rows)
    json_valid = 0
    tool_correct = 0
    args_correct = 0
    overall_correct = 0

    for row in rows:
        pred = try_parse_json(row.get("predict", ""))
        expect = row["expect"]

        if pred is not None:
            json_valid += 1

            # TODO 1: 比較 tool
            if pred.get("tool") == expect.get("tool"):
                tool_correct += 1

            # TODO 2: 比較 arguments
            if pred.get("arguments") == expect.get("arguments"):
                args_correct += 1

            # TODO 3: 全部正確才加 overall_correct
            if (
                pred.get("tool") == expect.get("tool")
                and pred.get("arguments") == expect.get("arguments")
            ):
                overall_correct += 1

    report = {
        "total": total,
        "json_valid_rate": round(json_valid / total, 4) if total else 0,
        "tool_correct_rate": round(tool_correct / total, 4) if total else 0,
        "args_correct_rate": round(args_correct / total, 4) if total else 0,
        "overall_accuracy": round(overall_correct / total, 4) if total else 0,
    }

    print(json.dumps(report, ensure_ascii=False, indent=2))


def main():
    rows = load_json(INPUT_FILE)
    evaluate(rows)


if __name__ == "__main__":
    main()