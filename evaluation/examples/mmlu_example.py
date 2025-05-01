"""
MMLU 데이터셋 평가 예제

이 예제는 MMLU 데이터셋을 사용하여 모델의 다중 선택형 문제 풀이 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

import sys
from evaluation.base.main import main

def run_mmlu_example():
    # 명령행 인자 설정
    sys.argv = [
        "mmlu_example.py",
        "--dataset", "mmlu",
        "--model", "gpt4o",
        "--model_version", "gpt-3.5-turbo",
        # "--model_version", "gpt-4o",
        # 기존 프롬프트
        "--base_system_prompt", "Choose the best answer among A, B, C, or D. Give only the letter.",
        "--base_user_prompt", "Question:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an AI proficient in logical reasoning. Carefully analyze problems step-by-step, openly assessing each provided option. Keep your evaluation concise, logically clear, and distinctly separate from your final formatted answer.",
        "--zera_user_prompt", "Evaluate the following multiple-choice question by logically analyzing each option. Briefly state why each option (A, B, C, or D) is correct or incorrect. After your analysis, clearly and concisely state your final answer as a single capital letter.\n\nExample:\n\nQuestion:  \nWhere is most chemical digestion and nutrient absorption completed in the human digestive system?\n\nChoices:  \nA. Stomach  \nB. Large intestine  \nC. Small intestine  \nD. Mouth  \n\nReasoning:  \n- A. The stomach primarily provides physical breakdown and initial protein digestion, rather than completing full chemical digestion and nutrient absorption.\n- B. The large intestine is primarily involved in water absorption and waste storage, not significant chemical digestion or nutrient absorption.\n- C. The small intestine is responsible for most chemical digestion and nearly all nutrient absorption, precisely matching the description.\n- D. The mouth performs initial mechanical breakdown and minor chemical digestion (carbohydrates), not significant overall digestion or nutrient absorption completion.\n\nFinal Answer: C\n\nNow follow the same approach with this question:\n\nQuestion:  \n[Insert Your Question Here]\n\nChoices:  \n[Insert Your Choices Here]",
        "--num_samples", "1000",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_mmlu_example() 