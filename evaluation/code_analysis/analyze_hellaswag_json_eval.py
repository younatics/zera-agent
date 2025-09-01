import json
import re
import sys
import glob
from typing import Any, Dict

def safe_json_load(path):
    with open(path, 'r') as f:
        data = f.read()
    data = data.replace('NaN', 'null')
    return json.loads(data)

def extract_hellaswag_official_choice(response: str) -> int:
    resp = response.strip().upper()
    matches = re.findall(r'[A-D]', resp)
    if matches:
        return ord(matches[-1]) - ord('A')
    matches = re.findall(r'[0-3]', resp)
    if matches:
        return int(matches[-1])
    return -1

def find_latest_hellaswag_json():
    files = glob.glob("HellaSwagEvaluator_*.json")
    if not files:
        return None
    files.sort(reverse=True)
    return files[0]

def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        input_path = find_latest_hellaswag_json()
        if not input_path:
            print("No HellaSwagEvaluator_*.json file found.")
            return
    data = safe_json_load(input_path)
    results = []
    original_correct = 0
    official_correct = 0
    for idx, sample in enumerate(data["samples"]):
        model_response = sample["model_response"]
        actual_answer = sample["actual_answer"]
        original_judgement = sample.get("is_correct", None)
        # Re-score using official method
        pred_idx = extract_hellaswag_official_choice(model_response)
        official_judgement = (pred_idx == actual_answer)
        if official_judgement:
            official_correct += 1
        if original_judgement:
            original_correct += 1
        results.append({
            "idx": idx,
            "model_response": model_response,
            "actual_answer": actual_answer,
            "pred_idx": pred_idx,
            "official_judgement": official_judgement,
            "original_judgement": original_judgement
        })
    total = len(results)
    original_accuracy = original_correct / total if total else 0.0
    official_accuracy = official_correct / total if total else 0.0
    print(f"\n[HellaSwag Official Scoring Method Re-scoring Results]")
    print(f"Total questions: {total}")
    print(f"Original score: {original_accuracy:.3f} (i.e., {original_correct}/{total})")
    print(f"Official scoring method re-scoring result: {official_accuracy:.3f} (i.e., {official_correct}/{total})")
    # Save results
    output_path = input_path.replace('.json', '_official_analysis.json')
    output = {
        "input_path": input_path,
        "total": total,
        "original_correct": original_correct,
        "original_accuracy": original_accuracy,
        "official_correct": official_correct,
        "official_accuracy": official_accuracy,
        "results": results
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Analysis results saved: {output_path}")

if __name__ == "__main__":
    main() 