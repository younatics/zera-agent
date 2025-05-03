"""
XSUM 데이터셋 평가 예제

이 예제는 XSUM 데이터셋을 사용하여 모델의 뉴스 요약 능력을 평가합니다.
기존 프롬프트와 제라 프롬프트를 동일한 샘플에 대해 비교 평가합니다.
"""

from evaluation.base.main import main
import sys

def run_xsum_example():
    # 명령줄 인자 설정
    sys.argv = [
        "xsum_example.py",
        "--dataset", "xsum",
        "--model", "gpt4o",
        "--model_version", "gpt-3.5-turbo",
        # 기존 프롬프트
        "--base_system_prompt", "Summarize the following article in one sentence:",
        "--base_user_prompt", "Article:",
        # 제라 프롬프트
        "--zera_system_prompt", "You are an AI assistant skilled in careful reading and logical reasoning. First, logically analyze the text provided, reasoning freely to pinpoint the single most newsworthy event. Then express this clearly as a concise, accurate, factual news headline.",
        "--zera_user_prompt", """Read the following text thoroughly and summarize its most newsworthy event in one concise, clear, headline-style sentence.

Examples:
Text: After prosecutors criticized the "shockingly light" jail sentence of the Olympic athlete convicted over his girlfriend's killing, they now request court approval to challenge the ruling.
Concise summary: Prosecutors in South Africa seek permission to appeal Oscar Pistorius' sentence, calling it "shockingly light."

Text: A rugby player's outstanding three-try performance against his former team secured Warrington's victory, moving them forward in the Challenge Cup competition.
Concise summary: Kevin Brown's hat-trick sends Warrington into Challenge Cup quarter-finals.

Text: Media playback is not supported on this device
Silverstone has been home to the race every year since 1987.
However, the British Racing Drivers' Club (BRDC), which owns the circuit, says it cannot afford to host the race unless a new deal is agreed.
"We have reached the tipping point," said BRDC chairman John Grant.
"We sustained losses of £2.8m in 2015 and £4.8m in 2016, and we expect to lose a similar amount this year."
Formula 1 owner Liberty Media said it regrets the BRDC's decision and will continue to negotiate in hope of preserving the British Grand Prix at Silverstone.
Concise summary: Future of British Grand Prix uncertain as Silverstone's owners activate break clause to cease hosting after 2019.

Now summarize the most newsworthy event in the text below:

Text: [Insert user's text here]
Concise summary:
        """,
        "--num_samples", "2",
        # 모델 파라미터
        # "--temperature", "0.2",  # 더 결정적인 응답을 위해 낮은 temperature 사용
        # "--top_p", "0.9"
    ]
    
    # 평가 실행
    main()

if __name__ == "__main__":
    run_xsum_example() 