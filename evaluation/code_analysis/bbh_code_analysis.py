import json
from evaluation.dataset_evaluator.bbh_evaluator import BBHEvaluator
import re


def main():
    # JSON 파일 경로
    json_path = "evaluation/code_analysis/BBHEvaluator_gpt-3.5-turbo_20250511_213353.json"
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    evaluator = BBHEvaluator("gpt4o", "gpt-3.5-turbo")
    total = data["total"]
    samples = data["samples"]

    match_count = 0
    mismatch_cases = []

    for idx, sample in enumerate(samples):
        model_response = sample["model_response"]
        actual_answer = sample["actual_answer"]
        original_is_correct = sample["is_correct"]

        # BBHEvaluator의 평가
        code_is_correct = evaluator.evaluate_response(model_response, {"answer": actual_answer})

        if code_is_correct == original_is_correct:
            match_count += 1
        else:
            mismatch_cases.append({
                "idx": idx,
                "question": sample["question"],
                "model_response": model_response,
                "actual_answer": actual_answer,
                "original_is_correct": original_is_correct,
                "code_is_correct": code_is_correct
            })

    print(f"총 {total}개 중 코드 평가와 기존 평가가 일치한 케이스: {match_count}개 ({match_count/total*100:.2f}%)")
    print(f"불일치 케이스: {len(mismatch_cases)}개")
    for case in mismatch_cases:
        print("="*40)
        print(f"Index: {case['idx']}")
        print(f"Question: {case['question']}")
        print(f"Model Response: {case['model_response']}")
        print(f"Actual Answer: {case['actual_answer']}")
        print(f"Original is_correct: {case['original_is_correct']}")
        print(f"Code is_correct: {case['code_is_correct']}")

    response_clean = 'THEREFORE, THE FINAL ANSWER IS (J) **TRIANGLE**.'
    print(re.findall(r'\(([A-Z0-9\.]+)\)', response_clean))

if __name__ == "__main__":
    main() 