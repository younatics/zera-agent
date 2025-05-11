"""
CNN/DailyMail 데이터셋 평가 예제

이 예제는 CNN/DailyMail 데이터셋을 사용하여 모델의 뉴스 요약 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

from evaluation.base.main import main
import sys

def run_cnn_dailymail_example(model="claude", model_version="claude-3-sonnet-20240229"):
    # 명령줄 인자 설정
    sys.argv = [
        "example.py",
        "--dataset", "cnn_dailymail",
        "--model", model,
        "--model_version", model_version,
        # 기존 프롬프트
        # "--base_system_prompt", "You are a summarization assistant. Summarize the following article in 2–3 sentences, focusing on the main idea.",
        # "--base_user_prompt", "Article:",
        # 제라 프롬프트
        "--zera_system_prompt", "Thoroughly analyze the given text to extract key information, underlying motivations, and broader implications. Reason logically to develop a comprehensive understanding by connecting different components. Once your analysis is complete, present the essential insights through a well-structured multi-sentence summary.",
        "--zera_user_prompt", """Based on the following article, summarize the crucial points in a multi-sentence format:

[ARTICLE TEXT]

Example:
- Central point capturing a key event or idea
- Important detail highlighting another core aspect  
- Relevant context or implication derived from the text

Prioritize conveying accurate insights through a natural flow, rather than rigidly adhering to formatting rules. Focus on clear logical communication of essential elements.

Reasoning:
- The task involves generating multi-sentence text summaries, requiring free-form reasoning balanced with structural guidance.
- The system prompt encourages comprehensive analysis, connecting different components, and deriving broader implications - the core reasoning process.
- The user prompt concisely frames the core task (summarizing article in multi-sentence format) with minimal formatting constraints.
- Including 1-3 examples illustrates the desired content (events, ideas, context) in a lightweight multi-sentence structure.
- By prioritizing "accurate insights" and "natural flow" over "rigid adherence", the prompt allows flexibility based on content needs.
- No arbitrary length limits are imposed, letting summaries adapt organically while still encouraging conciseness.
- The prompts maintain a healthy separation between reasoning (analysis and implications) and final presentation (clear multi-sentence summary).

This approach enables free-flowing reasoning tailored to each article, followed by a well-structured yet natural presentation of the derived insights. The focus is on conveying accurate and complete information coherently, rather than enforcing rigid formatting rules. Examples provide guidance while allowing model judgement based on content requirements.""",
        "--num_samples", "500",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_cnn_dailymail_example() 