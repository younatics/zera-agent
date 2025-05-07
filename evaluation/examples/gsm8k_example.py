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
        # "--base_system_prompt", "Provide the final answer prefixed with '####'.",
        # "--base_user_prompt", "Question:\n",
        # 제라 프롬프트
        "--zera_system_prompt", "You are a precise and logical AI assistant. Always reason step-by-step clearly and transparently, keeping your explanations concise. Present only your final answer on a separate line, formatted as '#### [answer]', after completing all required calculations.",
        "--zera_user_prompt", """Solve the given problem step-by-step with explicit inline calculations. Present your final answer clearly as indicated.

Example:
Question: Borgnine wants to see 1100 legs at the zoo. He has already seen 12 chimps, 8 lions, and 5 lizards. He is next headed to see the tarantulas. How many tarantulas does he need to see to meet his goal?

Answer:
He has seen 48 chimp legs because 12 x 4 = <<12*4=48>>48
He has seen 32 lion legs because 8 x 4 = <<8*4=32>>32
He has seen 20 lizard legs because 5 x 4 = <<5*4=20>>20
He has seen 100 total legs because 48 + 32 + 20 = <<48+32+20=100>>100
He must see 1000 more legs because 1100 - 100 = <<1100-100=1000>>1000
He needs to see 125 tarantulas because each has 8 legs and 1000 / 8 = <<1000/8=125>>125
#### 125""",
        "--num_samples", "1319",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_gsm8k_example() 