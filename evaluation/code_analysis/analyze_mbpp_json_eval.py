import json
import ast
import re
import logging
import tempfile
import subprocess
import difflib
import math
from typing import Any, Dict
import inspect

# 평가 함수 (mbpp_evaluator.py의 evaluate_response와 유사)
def run_safely(code: str, timeout=5):
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

def extract_code(response: str) -> str:
    # 코드블록이 있으면 내부만 추출
    code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    # 코드블록이 없으면 함수 정의부터 끝까지 추출 (fallback)
    func_def_match = re.search(r'(def [a-zA-Z_][a-zA-Z0-9_]*\s*\(.*)', response, re.DOTALL)
    if func_def_match:
        return func_def_match.group(1).strip()
    return response.strip()

def _find_best_func_name(code_block: str, target_func: str) -> str:
    func_names = re.findall(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_block)
    if not func_names:
        return None
    best = difflib.get_close_matches(target_func, func_names, n=1, cutoff=0.6)
    return best[0] if best else func_names[0]

def _auto_imports(code_block: str) -> str:
    imports = []
    if re.search(r'\bmath\.', code_block) and 'import math' not in code_block:
        imports.append('import math')
    if re.search(r'\bre\.', code_block) and 'import re' not in code_block:
        imports.append('import re')
    if re.search(r'\bcollections\.', code_block) and 'import collections' not in code_block:
        imports.append('import collections')
    if re.search(r'\bitertools\.', code_block) and 'import itertools' not in code_block:
        imports.append('import itertools')
    if re.search(r'\bfunctools\.', code_block) and 'import functools' not in code_block:
        imports.append('import functools')
    if re.search(r'\bheapq\.', code_block) and 'import heapq' not in code_block:
        imports.append('import heapq')
    if re.search(r'\bdatetime\.', code_block) and 'import datetime' not in code_block:
        imports.append('import datetime')
    return '\n'.join(imports) + ('\n' if imports else '') + code_block

def _wrap_lambda(code_block: str, func_name: str, gt_func: str) -> str:
    lambda_match = re.search(r'(\w+)\s*=\s*lambda', code_block)
    if lambda_match:
        name = lambda_match.group(1)
        sig_match = re.search(r'def '+re.escape(gt_func)+r'\((.*?)\)', code_block)
        sig = sig_match.group(1) if sig_match else 'x'
        return code_block + f"\ndef {gt_func}({sig}):\n    return {name}({sig})"
    return code_block

def _normalize_output(output):
    try:
        val = eval(output, {"__builtins__": {}})
        # float: 반올림
        if isinstance(val, float):
            return round(val, 6)
        # list/tuple/set: 모두 list로 변환
        if isinstance(val, (list, tuple, set)):
            return list(val)
        # dict: key, value 모두 str로 변환
        if isinstance(val, dict):
            return {str(k): str(v) for k, v in val.items()}
        # str: 소문자, 공백/쉼표/따옴표 제거
        if isinstance(val, str):
            return re.sub(r"[\s,'\"]", "", val.strip().lower())
        return val
    except Exception:
        # eval 실패 시 문자열로 처리
        return re.sub(r"[\s,'\"]", "", str(output).strip().lower())

def try_fix_args_and_run(func, args, kwargs=None):
    """
    func: 함수 객체
    args: 튜플
    kwargs: dict
    """
    if kwargs is None:
        kwargs = {}
    try:
        return func(*args, **kwargs)
    except TypeError as e:
        # 인자 개수 오류만 잡는다
        msg = str(e)
        if 'positional arguments but' in msg or 'missing' in msg or 'required positional argument' in msg:
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            # 인자 개수 맞추기
            n_expected = len(params)
            n_given = len(args)
            if n_given > n_expected:
                # 불필요한 인자 잘라내기
                new_args = args[:n_expected]
            else:
                # 부족하면 None으로 채우기
                new_args = list(args) + [None]*(n_expected-n_given)
            try:
                return func(*new_args, **kwargs)
            except Exception:
                return None
        else:
            return None
    except Exception:
        return None

def _extract_func_names(code_block: str):
    # 코드에서 함수명 모두 추출
    return re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_block)

def evaluate_response(response: str, ground_truth: Dict[str, Any]) -> (bool, str):
    try:
        # 1. 모델 답변 전체에서 import문 추출
        import_lines = "\n".join([line for line in response.splitlines() if line.strip().startswith("import") or line.strip().startswith("from ")])
        # 2. 코드블록 또는 함수 정의만 추출
        code_block = extract_code(response)
        # 정답 함수명 추출
        gt_func_match = re.search(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', ground_truth['code'])
        gt_func_name = gt_func_match.group(1) if gt_func_match else "unknown_function"
        # 함수명 자동 매핑
        best_func = _find_best_func_name(code_block, gt_func_name)
        if not best_func:
            code_block = _wrap_lambda(code_block, best_func, gt_func_name)
            best_func = gt_func_name
        # 함수명 치환 (테스트케이스에서 호출하는 함수명과 일치하도록)
        code_block = re.sub(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', f'def {gt_func_name}(', code_block, count=1)
        # 자동 import
        code_block = _auto_imports(code_block)
        # 테스트 케이스 실행
        try:
            test_cases = ast.literal_eval(ground_truth['test_list'])
        except Exception as e:
            return False, f"[테스트리스트파싱실패] {e} | test_list: {ground_truth['test_list']}"
        # 3. 합쳐서 실행
        full_code = import_lines + "\n" + code_block
        for test_case in test_cases:
            try:
                test_case = test_case.replace("assert ", "")
                call, expected = test_case.split("==")
                call = call.strip()
                expected = expected.strip()
                test_code = f"{full_code}\nprint({call})"
                stdout, stderr, success = run_safely(test_code, timeout=10)
                if not success:
                    return None, f"[채점불가:서브프로세스실패] {call} | stdout: {stdout} | stderr: {stderr}"
                actual_output = stdout.strip()
                # 유연 비교
                norm_actual = _normalize_output(actual_output)
                norm_expected = _normalize_output(expected)
                if isinstance(norm_actual, float) and isinstance(norm_expected, float):
                    if not math.isclose(norm_actual, norm_expected, rel_tol=1e-2, abs_tol=1e-2):
                        return False, f"[테스트실패] {call} -> {norm_actual} (예상: {norm_expected})"
                elif norm_actual != norm_expected:
                    return False, f"[테스트실패] {call} -> {norm_actual} (예상: {norm_expected})"
            except Exception as e:
                return None, f"[채점불가:테스트케이스실행실패] {test_case} | {e}"
        return True, "PASS"
    except Exception as e:
        return None, f"[채점불가:예외] {e}\nresponse: {response}\nground_truth: {ground_truth}"

def safe_json_load(path):
    with open(path, 'r') as f:
        data = f.read()
    # NaN -> null 치환
    data = data.replace('NaN', 'null')
    return json.loads(data)

def main():
    import sys
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        output_path = input_path.replace('.json', '_analysis.json')
    else:
        input_path = "evaluation/demotest/MBPPEvaluator_20250504_143421.json"
        output_path = "evaluation/demotest/MBPPEvaluator_20250504_143421_analysis.json"
    data = safe_json_load(input_path)
    results = []
    for idx, sample in enumerate(data["samples"]):
        response = sample["model_response"]
        ground_truth = sample["actual_answer"]
        is_correct, reason = evaluate_response(response, ground_truth)
        results.append({
            "idx": idx,
            "question": sample["question"],
            "model_response": response,
            "actual_answer": ground_truth,
            "is_correct_eval": is_correct,
            "reason": reason
        })
        if is_correct is None:
            print(f"[{idx}] -: {reason}")
        else:
            print(f"[{idx}] {'O' if is_correct else 'X'}: {reason}")
    # 정확도 계산 및 기록
    total = len(results)
    correct = sum(1 for r in results if r["is_correct_eval"] is True)
    wrong = sum(1 for r in results if r["is_correct_eval"] is False)
    unscorable = sum(1 for r in results if r["is_correct_eval"] is None)
    scorable_total = correct + wrong
    accuracy = correct / total if total > 0 else 0.0
    scorable_accuracy = correct / scorable_total if scorable_total > 0 else 0.0
    output = {
        "input_path": input_path,
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "unscorable": unscorable,
        "accuracy": accuracy,
        "scorable_total": scorable_total,
        "scorable_accuracy": scorable_accuracy,
        "results": results
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"분석 결과 저장: {output_path} (전체 정확도: {accuracy:.3f})")
    print(f"채점 가능한 문항 내 정확도: {scorable_accuracy:.3f} (채점 가능: {scorable_total}개 중 {correct}개 정답)")
    print(f"채점 불가(실행 실패 등): {unscorable}개")

if __name__ == "__main__":
    main() 