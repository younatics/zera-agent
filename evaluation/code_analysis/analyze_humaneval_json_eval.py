import json
import re
import tempfile
import subprocess
import os
import glob
import ast
from typing import Any, Dict

def run_safely(code: str, timeout=10):
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        temp_path = f.name
    try:
        result = subprocess.run(
            ["python3", temp_path],
            timeout=timeout,
            capture_output=True,
        )
        return result.stdout.decode(), result.stderr.decode(), result.returncode == 0
    except subprocess.TimeoutExpired:
        return "Timeout", "", False
    finally:
        os.remove(temp_path)

def extract_code(response: str) -> str:
    code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    func_def_match = re.search(r'(def [a-zA-Z_][a-zA-Z0-9_]*\s*\(.*)', response, re.DOTALL)
    if func_def_match:
        return func_def_match.group(1).strip()
    return response.strip()

def safe_json_load(path):
    with open(path, 'r') as f:
        data = f.read()
    data = data.replace('NaN', 'null')
    return json.loads(data)

def evaluate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    response = sample["model_response"]
    actual_answer = sample["actual_answer"]
    code_block = extract_code(response)
    # Extract test cases (should be a string of python code with asserts)
    test_cases = actual_answer.get("test_cases", "")
    entry_point = actual_answer.get("entry_point", None)
    # Compose code to run: model code + test cases
    code_to_run = code_block + "\n" + test_cases
    # Try to run
    try:
        stdout, stderr, exec_success = run_safely(code_to_run, timeout=10)
        if not exec_success:
            reason = f"[실행실패] stderr: {stderr.strip()} stdout: {stdout.strip()}"
            return {"is_correct_eval": False, "exec_success": False, "reason": reason}
        # If all asserts pass, it's correct
        if 'assert' in test_cases:
            if 'AssertionError' in stderr or 'Traceback' in stderr:
                reason = f"[테스트실패] stderr: {stderr.strip()}"
                return {"is_correct_eval": False, "exec_success": True, "reason": reason}
        return {"is_correct_eval": True, "exec_success": True, "reason": "PASS"}
    except Exception as e:
        return {"is_correct_eval": False, "exec_success": False, "reason": f"[예외] {e}"}

def find_latest_humaneval_json():
    files = glob.glob("HumanEvalEvaluator_*.json")
    if not files:
        return None
    files.sort(reverse=True)
    return files[0]

def main():
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        # Find latest HumanEvalEvaluator_*.json in current dir
        input_path = find_latest_humaneval_json()
        if not input_path:
            print("No HumanEvalEvaluator_*.json file found.")
            return
    data = safe_json_load(input_path)
    results = []
    for idx, sample in enumerate(data["samples"]):
        res = evaluate_sample(sample)
        results.append({
            "idx": idx,
            "question": sample["question"],
            "model_response": sample["model_response"],
            "actual_answer": sample["actual_answer"],
            **res
        })
        print(f"[{idx}] {'O' if res['is_correct_eval'] else 'X'} | 실행: {'O' if res['exec_success'] else 'X'} | {res['reason']}")
    # 통계
    total = len(results)
    correct = sum(1 for r in results if r["is_correct_eval"])
    exec_success = [r for r in results if r["exec_success"]]
    exec_success_total = len(exec_success)
    exec_success_correct = sum(1 for r in exec_success if r["is_correct_eval"])
    exec_success_fail = exec_success_total - exec_success_correct
    # 실패 예시
    fail_exec = [r for r in results if not r["exec_success"]][:3]
    fail_test = [r for r in exec_success if not r["is_correct_eval"]][:3]
    # 저장
    output_path = input_path.replace('.json', '_analysis.json')
    output = {
        "input_path": input_path,
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "exec_success_total": exec_success_total,
        "exec_success_correct": exec_success_correct,
        "exec_success_fail": exec_success_fail,
        "exec_success_accuracy": exec_success_correct / exec_success_total if exec_success_total else 0.0,
        "results": results
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n분석 결과 저장: {output_path}")
    print(f"전체: {total}, 정답: {correct}, 정확도: {output['accuracy']:.3f}")
    print(f"실행 성공: {exec_success_total}, 실행 성공 중 정답: {exec_success_correct}, 실행 성공 중 오답: {exec_success_fail}, 실행 성공 내 정확도: {output['exec_success_accuracy']:.3f}")
    if fail_exec:
        print(f"\n[실행 실패 예시]")
        for r in fail_exec:
            print(f"  - idx {r['idx']}: {r['reason']}")
    if fail_test:
        print(f"\n[테스트 실패 예시]")
        for r in fail_test:
            print(f"  - idx {r['idx']}: {r['reason']}")

if __name__ == "__main__":
    main() 