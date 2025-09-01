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

# Evaluation function (similar to evaluate_response in mbpp_evaluator.py)
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
    # If code block exists, extract only the content inside
    code_match = re.search(r"```(?:python)?(.*?)```", response, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    # If no code block, extract from function definition to end (fallback)
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
        # float: round
        if isinstance(val, float):
            if val.is_integer():
                return int(val)
            return round(val, 6)
        if isinstance(val, int):
            return val
        # list/tuple/set: convert all to list, sort for comparison
        if isinstance(val, (list, tuple, set)):
            try:
                return sorted(list(val))
            except Exception:
                return list(val)
        # dict: sort by key, convert value to str
        if isinstance(val, dict):
            return {str(k): str(v) for k, v in sorted(val.items())}
        # str: lowercase, remove spaces/commas/quotes
        if isinstance(val, str):
            return re.sub(r"[\s,'\"]", "", val.strip().lower())
        return val
    except Exception:
        # If eval fails, process as string
        return re.sub(r"[\s,'\"]", "", str(output).strip().lower())

def _is_similar(a, b, threshold=0.7):
    if not isinstance(a, str):
        a = str(a)
    if not isinstance(b, str):
        b = str(b)
    seq = difflib.SequenceMatcher(None, a, b)
    return seq.ratio() >= threshold or (a in b) or (b in a)

def try_fix_args_and_run(func, args, kwargs=None):
    """
    func: function object
    args: tuple
    kwargs: dict
    """
    if kwargs is None:
        kwargs = {}
    try:
        return func(*args, **kwargs)
    except TypeError as e:
        # Only catch argument count errors
        msg = str(e)
        if 'positional arguments but' in msg or 'missing' in msg or 'required positional argument' in msg:
            sig = inspect.signature(func)
            params = list(sig.parameters.values())
            # Match argument count
            n_expected = len(params)
            n_given = len(args)
            if n_given > n_expected:
                # Cut unnecessary arguments
                new_args = args[:n_expected]
            else:
                # Fill with None if insufficient
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
    # Extract all function names from code
    return re.findall(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_block)

def evaluate_response(response: str, ground_truth: Dict[str, Any]) -> (bool, str):
    try:
        # 1. Extract import statements from entire model response
        import_lines = "\n".join([line for line in response.splitlines() if line.strip().startswith("import") or line.strip().startswith("from ")])
        # 2. Extract only code block or function definition
        code_block = extract_code(response)
        # Extract ground truth function name
        gt_func_match = re.search(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', ground_truth['code'])
        gt_func_name = gt_func_match.group(1) if gt_func_match else "unknown_function"
        # Auto function name mapping
        best_func = _find_best_func_name(code_block, gt_func_name)
        if not best_func:
            code_block = _wrap_lambda(code_block, best_func, gt_func_name)
            best_func = gt_func_name
        # Function name replacement (to match function name called in test cases)
        code_block = re.sub(r'def ([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', f'def {gt_func_name}(', code_block, count=1)
        # Auto import
        code_block = _auto_imports(code_block)
        # Execute test cases
        try:
            test_cases = ast.literal_eval(ground_truth['test_list'])
        except Exception as e:
            return False, f"[Test List Parsing Failed] {e} | test_list: {ground_truth['test_list']}"
        # 3. Combine and execute
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
                    return None, f"[Cannot Grade: Subprocess Failed] {call} | stdout: {stdout} | stderr: {stderr}"
                actual_output = stdout.strip()
                # Flexible comparison
                norm_actual = _normalize_output(actual_output)
                norm_expected = _normalize_output(expected)
                # float comparison
                if isinstance(norm_actual, float) and isinstance(norm_expected, float):
                    if not math.isclose(norm_actual, norm_expected, rel_tol=1e-2, abs_tol=1e-2):
                        return False, f"[Test Failed] {call} -> {norm_actual} (Expected: {norm_expected})"
                # list/tuple/set: convert and compare
                elif isinstance(norm_actual, list) and isinstance(norm_expected, list):
                    try:
                        if sorted(norm_actual) != sorted(norm_expected):
                            # Also compare as string
                            if str(norm_actual) != str(norm_expected):
                                return False, f"[Test Failed] {call} -> {norm_actual} (Expected: {norm_expected})"
                    except Exception:
                        if norm_actual != norm_expected:
                            return False, f"[Test Failed] {call} -> {norm_actual} (Expected: {norm_expected})"
                # dict: convert and compare
                elif isinstance(norm_actual, dict) and isinstance(norm_expected, dict):
                    if norm_actual != norm_expected:
                        # Also compare as string
                        if str(norm_actual) != str(norm_expected):
                            return False, f"[Test Failed] {call} -> {norm_actual} (Expected: {norm_expected})"
                # aggressive numeric type casting comparison
                elif _aggressive_number_compare(norm_actual, norm_expected):
                    pass  # Accept as correct
                # If types differ, try converting for comparison
                elif type(norm_actual) != type(norm_expected):
                    # str <-> list conversion
                    try:
                        if isinstance(norm_actual, str) and isinstance(norm_expected, (list, tuple, set)):
                            if sorted(list(norm_actual)) != sorted(list(norm_expected)):
                                return False, f"[Test Failed] {call} -> {norm_actual} (Expected: {norm_expected})"
                        elif isinstance(norm_expected, str) and isinstance(norm_actual, (list, tuple, set)):
                            if sorted(list(norm_expected)) != sorted(list(norm_actual)):
                                return False, f"[Test Failed] {call} -> {norm_actual} (Expected: {norm_expected})"
                        elif isinstance(norm_actual, (int, float)) and isinstance(norm_expected, (int, float)):
                            if not math.isclose(float(norm_actual), float(norm_expected), rel_tol=1e-2, abs_tol=1e-2):
                                return False, f"[Test Failed] {call} -> {norm_actual} (Expected: {norm_expected})"
                        else:
                            if str(norm_actual) != str(norm_expected):
                                return False, f"[Test Failed] {call} -> {norm_actual} (Expected: {norm_expected})"
                    except Exception:
                        if str(norm_actual) != str(norm_expected):
                            return False, f"[Test Failed] {call} -> {norm_actual} (Expected: {norm_expected})"
                # str similarity/partial match comparison
                elif isinstance(norm_actual, str) and isinstance(norm_expected, str):
                    if not _is_similar(norm_actual, norm_expected, threshold=0.7):
                        return False, f"[Test Failed] {call} -> {norm_actual} (Expected: {norm_expected})"
                else:
                    if norm_actual != norm_expected:
                        return False, f"[Test Failed] {call} -> {norm_actual} (Expected: {norm_expected})"
            except Exception as e:
                return None, f"[Cannot Grade: Test Case Execution Failed] {test_case} | {e}"
        return True, "PASS"
    except Exception as e:
        return None, f"[Cannot Grade: Exception] {e}\nresponse: {response}\nground_truth: {ground_truth}"

def safe_json_load(path):
    with open(path, 'r') as f:
        data = f.read()
    # Replace NaN -> null
    data = data.replace('NaN', 'null')
    return json.loads(data)

def _aggressive_number_compare(a, b):
    """
    More aggressively attempt numeric type comparison (2.0 vs 2, 2 vs '2', '2.0' vs 2, etc.)
    """
    try:
        # Try converting both to numeric types (including str)
        a_num = float(a) if not isinstance(a, (int, float)) else a
        b_num = float(b) if not isinstance(b, (int, float)) else b
        if math.isclose(a_num, b_num, rel_tol=1e-2, abs_tol=1e-2):
            return True
    except Exception:
        pass
    try:
        # Compare by converting int/float <-> str
        if str(a) == str(b):
            return True
    except Exception:
        pass
    return False

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
    # Calculate and record accuracy
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
    print(f"Analysis results saved: {output_path} (Overall accuracy: {accuracy:.3f})")
    print(f"Accuracy within gradable questions: {scorable_accuracy:.3f} (Gradable: {correct} correct out of {scorable_total})")
    print(f"Ungradable (execution failure, etc.): {unscorable}")

if __name__ == "__main__":
    main() 