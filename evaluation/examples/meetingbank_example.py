"""
MeetingBank Dataset Evaluation Example

This example evaluates the model's meeting summarization ability using the MeetingBank dataset.
It compares the existing prompt and Zera prompt on the same samples.
"""

from evaluation.base.main import main
import sys

def run_meetingbank_example(model="claude", model_version="claude-3-sonnet-20240229"):
    # Set command line arguments
    sys.argv = [
        "example.py",
        "--dataset", "meetingbank",
        "--model", model,
        "--model_version", model_version,
        # Existing prompt
        # "--base_system_prompt", "You are a meeting summarization assistant. Summarize the following meeting transcript in 2-3 sentences, focusing on the main points and key decisions.",
        # "--base_user_prompt", "Meeting Transcript:",
        # Zera prompt
        "--zera_system_prompt", "You are a helpful and informative AI model. Answer questions in a clear, concise, and well-reasoned manner, while enforcing output format only when presenting the final answer.",
        "--zera_user_prompt", """User: I have a question about defunding the police. [Insert Question Here]
Assistant: [Insert Answer Here]

In this scenario, the user prompt includes the question and the assistant's response. No examples are needed as the question itself provides a clear context for the model to generate an appropriate response.

I recommend considering the following steps when optimizing prompts for other tasks:

1. Analyze the reasoning and output requirements of the task.
2. Identify the weaknesses and strengths in the original prompt and previous iterations.
3. Apply concise improvements that enhance clarity, task alignment, and output structure without increasing prompt complexity.
4. Validate the new prompts for brevity, clear separation of reasoning and formatting, minimal formatting requirements, and lack of redundancy or overlong instructions.
5. Repeat the process as needed to further refine the prompts and improve performance.""",
        "--num_samples", "1000",
        # # Model parameters
        # "--temperature", "0.2",  # Use low temperature for more deterministic responses
        # "--top_p", "0.9"
    ]
    
    # Execute evaluation
    main()

if __name__ == "__main__":
    run_meetingbank_example() 