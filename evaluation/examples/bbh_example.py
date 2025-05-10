"""
BBH (Big-Bench Hard) 데이터셋 평가 예제

이 예제는 BBH 데이터셋을 사용하여 모델의 다양한 태스크 수행 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_bbh_example(model="gpt4o", model_version="gpt-3.5-turbo", bbh_category=None):
    # 명령행 인자 설정
    sys.argv = [
        "bbh_example.py",
        "--dataset", "bbh",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        # "--base_system_prompt", "Answer the following question.",
        # "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an AI assistant ready to help.",
        "--zera_user_prompt", """Please determine the geometric shape drawn by the provided SVG path element. Encourage logical reasoning to identify the shape accurately.""",
        "--num_samples", "100"
    ]
    if bbh_category:
        sys.argv += ["--bbh_category", bbh_category]
    # 평가 실행
    main()

if __name__ == "__main__":
    run_bbh_example(bbh_category="Geometry")