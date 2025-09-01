from ..api_client import Model
import os
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Define test question
TEST_QUESTION = "Hello! Who are you?"

# Set API keys
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
solar_client = OpenAI(
    api_key=os.getenv("SOLAR_API_KEY"),
    base_url="https://api.upstage.ai/v1"
)

def test_model_initialization():
    print("\n=== Model Initialization Test ===")
    
    # GPT model test
    try:
        model = Model("gpt4o")
        print("GPT model initialization successful:", model.name)
    except Exception as e:
        print("GPT model initialization failed:", str(e))
    
    # Claude model test
    try:
        model = Model("claude")
        print("Claude model initialization successful:", model.name)
    except Exception as e:
        print("Claude model initialization failed:", str(e))
    
    # Solar Pro model test
    try:
        model = Model("solar")
        print("Solar Pro model initialization successful:", model.name)
    except Exception as e:
        print("Solar Pro model initialization failed:", str(e))
    
    # Invalid model name test
    try:
        model = Model("invalid_model")
    except Exception as e:
        print("Invalid model name test successful:", str(e))

def test_available_models():
    print("\n=== Available Models List Test ===")
    models = Model.get_available_models()
    print("Available models:", models)

def test_model_responses():
    print("\n=== Model Question Test ===")
    
    # GPT model test
    print("\n[GPT Model Test]")
    gpt_model = Model("gpt4o")
    gpt_response = gpt_model.ask(TEST_QUESTION)
    print("GPT response:", gpt_response)
    
    # Claude model test
    print("\n[Claude Model Test]")
    claude_model = Model("claude")
    claude_response = claude_model.ask(TEST_QUESTION)
    print("Claude response:", claude_response)
    
    # Solar Pro model test
    print("\n[Solar Pro Model Test]")
    solar_model = Model("solar")
    solar_response = solar_model.ask(TEST_QUESTION)
    print("Solar Pro response:", solar_response)

if __name__ == "__main__":
    print("API client test started")
    test_model_initialization()
    test_available_models()
    test_model_responses()
    print("\nTest completed") 