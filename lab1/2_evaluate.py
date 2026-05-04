import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jsonschema import validate as jsonschema_validate
from jsonschema.exceptions import ValidationError

from common.prompts import system_prompt
from common.tool_schema import TOOLS
from common.utils import extract_json_block


LAB_DIR = Path(__file__).parent
INPUT_FILE = LAB_DIR / "baseline_outputs.json"
TOOL_INDEX = {t["name"]: t for t in TOOLS}


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==============================================================================
# Step 1: 驗證 tool_call 格式
# ==============================================================================

def validate_tool_call(tool_call: dict) -> tuple[bool, str | None]:
    """
    驗證 LLM 輸出的 tool_call 是否合法
    
    依序執行以下檢查：
      1. tool_call 是否為 dict
      2. type 是否為 "tool_call"
      3. name 是否是已知工具
      4. arguments 是否為 dict
      5. arguments 是否符合該工具的 JSON Schema

    Returns:
        (True, None)          → 驗證通過
        (False, "錯誤說明")   → 驗證失敗
    """
    if not isinstance(tool_call, dict):
        return False, "tool_call_not_object"

    if tool_call.get("type") != "tool_call":
        return False, "type_must_be_tool_call"

    name = tool_call.get("name")
    if name not in TOOL_INDEX:
        return False, f"unknown_tool:{name}"

    args = tool_call.get("arguments")
    if not isinstance(args, dict):
        return False, "arguments_must_be_object"

    schema = TOOL_INDEX[name]["parameters"]
    try:
        jsonschema_validate(instance=args, schema=schema)
    except ValidationError as e:
        return False, f"schema_error:{e.message}"

    return True, None


# ==============================================================================
# Step 2: 評估指標函式
# ==============================================================================

def tool_selection_correct(pred_tool: str | None, expect: dict) -> bool:
    """
    判斷工具選擇是否正確

    判斷邏輯：
      - expect 有 "tool" → 比對 pred_tool 是否等於 expect["tool"]
      - expect 沒有 "tool" → pred_tool 應該是 None（不呼叫任何工具）

    Args:
        pred_tool: LLM 預測的工具名稱（或 None）
        expect:    測試案例的 expect 欄位

    Returns:
        True 表示選擇正確

    範例：
        tool_selection_correct("get_order_status", {"tool": "get_order_status"}) → True
        tool_selection_correct(None, {"should_ask_clarification": True})         → True
        tool_selection_correct("track_shipment", {"tool": "get_order_status"})   → False
    """
    if "tool" in expect:
        return pred_tool == expect["tool"]
    return pred_tool is None


def args_exact_match(pred_args: dict | None, expect: dict) -> bool:
    """
    判斷預測的參數是否與期望完全相符

    判斷邏輯：
      - expect 有 "arguments" → 直接比對 pred_args == expect["arguments"]
      - expect 沒有 "arguments" → pred_args 應該是 None

    Args:
        pred_args: LLM 預測的參數字典（或 None）
        expect:    測試案例的 expect 欄位

    Returns:
        True 表示參數完全相符

    範例：
        args_exact_match({"order_id": "A123456789"}, {"arguments": {"order_id": "A123456789"}}) → True
        args_exact_match({"order_id": "A999999999"}, {"arguments": {"order_id": "A123456789"}}) → False
    """
    if "arguments" not in expect:
        return pred_args is None
    return pred_args == expect["arguments"]


# ==============================================================================
# Step 3: 對單一案例執行 LLM 推論
# ==============================================================================

def run_one(case: dict) -> dict:
    """
    對單一測試案例執行 LLM 推論，回傳預測結果

    流程：
      1. 組合 messages（system prompt + 測試案例的 messages）
      2. 呼叫 LLM
      3. 提取 JSON（tool_call）
      4. 驗證格式

    Returns:
        {
            "raw":       str,        # LLM 原始輸出（供除錯用）
            "tool":      str | None, # 預測的工具名稱
            "arguments": dict | None,# 預測的參數
            "valid":     bool,       # tool_call 格式是否合法
            "error":     str | None  # 驗證錯誤訊息
        }
    """
    messages = [{"role": "system", "content": system_prompt()}] + case["messages"]
    out = case["predict"]
    print(out)
    pred = {
        "raw": out,
        "tool": None,
        "arguments": None,
        "valid": False,
        "error": None,
    }

    try:
        tool_call = extract_json_block(out)
        if tool_call is None:
            return pred
        ok, err = validate_tool_call(tool_call)
        pred["tool"]      = tool_call.get("name")
        pred["arguments"] = tool_call.get("arguments") or tool_call.get("parameters")
        pred["valid"]     = ok
        pred["error"]     = err
    except Exception as e:
        pred["error"] = f"json_parse_error:{e}"

    return pred



def main():
    cases = load_json(INPUT_FILE)

    # --------------------------------------------------------------------------
    # 對每個案例執行推論並計算指標
    # --------------------------------------------------------------------------
    rows = []
    for c in cases:
        pred = run_one(c)
        expect = c["expect"]

        row = {
            "id": c["id"],
            "valid": pred["valid"],
            "pred_tool": pred["tool"],
            "pred_args": pred["arguments"],
            "expect": expect,
            "tool_ok": tool_selection_correct(pred["tool"], expect),
            "args_ok": (
                args_exact_match(pred["arguments"], expect)
                if "arguments" in expect
                else None
            ),
        }
        rows.append(row)

    # --------------------------------------------------------------------------
    # 計算彙總統計
    # --------------------------------------------------------------------------
    n = len(rows)

    valid_rate = sum(1 for r in rows if r["valid"]) / n
    tool_acc = sum(1 for r in rows if r["tool_ok"]) / n
    args_cases = [r for r in rows if r["args_ok"] is not None]
    args_exact = (
        sum(1 for r in args_cases if r["args_ok"]) / len(args_cases)
        if args_cases else 0.0
    )

    summary = {
        "n": n,
        "valid_rate": round(valid_rate, 4),
        "tool_acc": round(tool_acc, 4),
        "args_exact": round(args_exact, 4),
    }

    # --------------------------------------------------------------------------
    # 輸出報告
    # --------------------------------------------------------------------------
    report_path = "eval_report.json"

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, ensure_ascii=False, indent=2)

    # --------------------------------------------------------------------------
    # 顯示結果
    # --------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("評估結果")
    print("=" * 60)
    print(f"總案例數：{n}")
    print(f"格式合法率     (valid_rate)：{valid_rate:.1%}")
    print(f"工具選擇準確率   (tool_acc)：{tool_acc:.1%}")
    print(f"參數完全相符率 (args_exact)：{args_exact:.1%}")
    print("=" * 60)
    print(f"詳細報告：{report_path}")

    # 顯示失敗案例（最多顯示 5 個）
    failed = [r for r in rows if not r["tool_ok"] or r["args_ok"] is False]
    if failed:
        print(f"\n失敗案例（{len(failed)} 個）：")
        for r in failed[:5]:
            print(f"  [{r['id']}] tool_ok={r['tool_ok']}, args_ok={r['args_ok']}")
            print(f"    預測：tool={r['pred_tool']}, args={r['pred_args']}")
            print(f"    期望：{r['expect']}")


if __name__ == "__main__":
    main()