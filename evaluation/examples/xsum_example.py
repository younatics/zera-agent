"""
XSUM Dataset Evaluation Example

This example evaluates the model's news summarization ability using the XSUM dataset.
It compares existing prompts and Zera prompts on the same samples.
"""

from evaluation.base.main import main
import sys

def run_xsum_example(model="gpt4o", model_version="gpt-3.5-turbo"):
    # Set command line arguments
    sys.argv = [
        "xsum_example.py",
        "--dataset", "xsum",
        "--model", model,
        "--model_version", model_version,
        # Base prompts
        "--base_system_prompt", "Summarize the following article in one sentence:",
        "--base_user_prompt", "Article:",
        # Zera prompts
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
        "--num_samples", "1000",
        # Model parameters
        # "--temperature", "0.2",  # Use low temperature for more deterministic responses
        # "--top_p", "0.9"
    ]
    
    # Execute evaluation
    main()

if __name__ == "__main__":
    run_xsum_example() 