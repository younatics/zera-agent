from evaluation.dataset_evaluator.bbh_evaluator import BBHEvaluator
import re

def main():
    evaluator = BBHEvaluator("gpt4o", "gpt-3.5-turbo")
    response = (
        '- "M": Moves the pen to point (45.10, 9.67) without drawing.\n'
        '- "L": Draws a straight line from (45.10, 9.67) to (15.22, 33.95).\n'
        '- "L": Continues drawing a straight line from (15.22, 33.95) to (77.94, 37.48).\n'
        '- "L": Draws another straight line from (77.94, 37.48) back to the starting point (45.10, 9.67).\n'
        '- "Z": Closes the path by connecting the last point to the initial starting point.\n\n'
        'Explicitly counted elements:\n'
        '- Distinct points: 4 (45.10, 9.67), (15.22, 33.95), (77.94, 37.48), (45.10, 9.67)\n'
        '- Line segments: 3\n'
        '- Arcs: 0\n\n'
        'Based on the analysis, the shape formed by the given SVG path is a **triangle** since it consists of 3 line segments connecting the points in a closed path. \n\n'
        'Therefore, the final answer is (J) **triangle**.'
    )
    actual_answer = "(J)"
    response_clean = response.strip().upper()
    print("[TEST DEBUG] response_clean:", response_clean)
    matches = re.findall(r'\(([A-Z0-9\.]+)\)', response_clean)
    print("[TEST DEBUG] 괄호 매치:", matches)
    if matches:
        model_answer = matches[-1]
        print("[TEST DEBUG] 최종 model_answer:", model_answer)
    else:
        print("[TEST DEBUG] 괄호 매치 없음")
    # 기존 평가 로직도 실행
    result = evaluator.evaluate_response(response, {"answer": actual_answer})
    print(f"evaluate_response 결과: {result}")

if __name__ == "__main__":
    main() 