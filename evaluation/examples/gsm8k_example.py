"""
GSM8K 데이터셋 평가 예제

이 예제는 GSM8K 데이터셋을 사용하여 모델의 수학 문제 풀이 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_gsm8k_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # 명령행 인자 설정
    sys.argv = [
        "gsm8k_example.py",
        "--dataset", "gsm8k",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        "--base_system_prompt", "Provide the final answer prefixed with '####'.",
        "--base_user_prompt", "Question:\n",
        # 제라 프롬프트
        "--zera_system_prompt", "You are a logical reasoning assistant. First reason through the problem naturally and clearly—ignoring formatting. Only at the final stage, concisely summarize critical numeric calculations using the designated \"<<calculation=result>>\" notation, and clearly report your final numeric answer.",
        "--zera_user_prompt", "Solve the following problem step-by-step with clear, logical reasoning. Afterward, briefly present each critical calculation step explicitly using the \"<<calculation=result>>\" notation, concluding with your final numeric answer clearly marked after \"####\".\n\nExample:\nQuestion: Sara buys 4 bouquets of roses, each bouquet has 12 roses. She gives away 9 roses. How many roses does Sara have left?\n\nReasoning:\nSara first buys a total of 4 bouquets * 12 roses each = 48 roses. Then she gives away 9 roses, leaving her with 48 roses - 9 roses = 39 roses.\n\nCalculations:\nTotal roses bought: 4 * 12 = <<4*12=48>>  \nRoses remaining: 48 - 9 = <<48-9=39>>\n\n#### 39\n\nNow solve this problem:\n",
        "--num_samples", "1319",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_gsm8k_example() 