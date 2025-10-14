import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.common.api_client import Model, create_messages


class DummyChatCompletions:
    def __init__(self, response):
        self._response = response
        self.last_request = None

    def create(self, **kwargs):
        self.last_request = kwargs
        return self._response


class DummyClient:
    def __init__(self, response):
        self.chat = SimpleNamespace(completions=DummyChatCompletions(response))


def build_openai_response(text, prompt_tokens=0, completion_tokens=0, total_tokens=None):
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens if total_tokens is not None else prompt_tokens + completion_tokens,
    )
    message = SimpleNamespace(content=text)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice], usage=usage)


def test_create_messages_handles_empty_segments():
    messages = create_messages("What is new?", system_prompt="system", user_prompt="")

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "system"
    assert messages[1]["content"] == "What is new?"


def test_model_ask_returns_metadata_and_uses_client():
    response = build_openai_response("hello", prompt_tokens=1000, completion_tokens=500)
    dummy_client = DummyClient(response)
    model = Model("gpt4o", client=dummy_client)

    answer, metadata = model.ask("Question?", system_prompt="System", user_prompt="Be kind")

    assert answer == "hello"
    assert metadata["input_tokens"] == 1000
    assert metadata["output_tokens"] == 500
    assert metadata["total_tokens"] == 1500
    expected_cost = (
        (1000 / 1000) * Model.token_prices["gpt4o"]["input_price_per_1k"]
        + (500 / 1000) * Model.token_prices["gpt4o"]["output_price_per_1k"]
    )
    assert metadata["cost"] == pytest.approx(expected_cost)
    assert metadata["model"] == model.model_id

    sent_messages = dummy_client.chat.completions.last_request["messages"]
    assert sent_messages[0]["content"] == "System"
    assert "Be kind" in sent_messages[1]["content"]


def test_ask_simple_delegates_to_ask():
    response = build_openai_response("result")
    model = Model("gpt4o", client=DummyClient(response))

    assert model.ask_simple("Question?") == "result"


def test_temperature_validation():
    model = Model("gpt4o", client=DummyClient(build_openai_response("answer")))

    with pytest.raises(ValueError):
        model.set_temperature(1.5)

    # Valid assignment should not raise
    model.set_temperature(0.2)


def test_get_model_info_contains_provider():
    info = Model.get_model_info("gpt4o")
    assert info["provider"] == "openai"
