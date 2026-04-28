from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional, Tuple


def _load_openai_client():
    try:
        from openai import OpenAI
    except ModuleNotFoundError as exc:
        raise ImportError(
            "The 'openai' package is required to use OpenAI-compatible models. "
            "Install project dependencies before creating this model client."
        ) from exc
    return OpenAI


def _load_anthropic_client():
    try:
        from anthropic import Anthropic
    except ModuleNotFoundError as exc:
        raise ImportError(
            "The 'anthropic' package is required to use Claude models. "
            "Install project dependencies before creating this model client."
        ) from exc
    return Anthropic


def create_messages(question: str, system_prompt: str, user_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{user_prompt}\n\n{question}"},
    ]


class Model:
    # Model-specific pricing information (USD per 1K input/output tokens).
    token_prices = {
        "solar": {"input_price_per_1k": 0.0002, "output_price_per_1k": 0.0004},
        "gpt4o": {"input_price_per_1k": 0.005, "output_price_per_1k": 0.015},
        "claude": {"input_price_per_1k": 0.003, "output_price_per_1k": 0.015},
        "local1": {"input_price_per_1k": 0.0, "output_price_per_1k": 0.0},
        "local2": {"input_price_per_1k": 0.0, "output_price_per_1k": 0.0},
        "solar_strawberry": {"input_price_per_1k": 0.0002, "output_price_per_1k": 0.0004},
    }

    model_info = {
        "solar": {
            "name": "Solar",
            "description": "Upstage's Solar model",
            "default_version": "solar-pro",
            "base_url": "https://api.upstage.ai/v1",
        },
        "gpt4o": {
            "name": "GPT-4",
            "description": "OpenAI's GPT-4 model",
            "default_version": "gpt-4o",
            "base_url": None,
        },
        "claude": {
            "name": "Claude",
            "description": "Anthropic's Claude 3.5 Sonnet",
            "default_version": "claude-3-5-sonnet-20240620",
            "base_url": None,
        },
        "local1": {
            "name": "Local Model 1",
            "description": "First Mistral model running on local server",
            "default_version": "/data/project/private/kyle/hf_models/Meta-Llama-3-70B-Instruct",
            "base_url": "http://localhost:8001/v1",
        },
        "local2": {
            "name": "Local Model 2",
            "description": "Second Mistral model running on local server",
            "default_version": "/data/project/private/kyle/hf_models/Mistral-7B-Instruct-v0.3",
            "base_url": "http://localhost:8002/v1",
        },
        "solar_strawberry": {
            "name": "Solar-Strawberry",
            "description": "Upstage's Solar-Strawberry model",
            "default_version": "Solar-Strawberry",
            "base_url": "https://r-api.toy.x.upstage.ai/v1/",
        },
    }

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Any]]:
        return cls.model_info.get(model_name)

    @classmethod
    def get_all_model_info(cls) -> Dict[str, Dict[str, Any]]:
        return cls.model_info

    @classmethod
    def get_available_models(cls) -> list[str]:
        return list(cls.model_info.keys())

    def __init__(self, model_name: str, version: Optional[str] = None, client: Any = None):
        if model_name not in self.model_info:
            raise ValueError(
                f"Model {model_name} not found. Available models: {list(self.model_info.keys())}"
            )

        self.name = model_name
        self.model_id = version or self.model_info[model_name]["default_version"]
        self.system_prompt = None
        self.user_prompt = None
        self.temperature = None
        self.top_p = None
        self.client = client or self._create_client(model_name)
        self.handler = self._create_handler()

    def _create_client(self, model_name: str) -> Any:
        if model_name == "claude":
            Anthropic = _load_anthropic_client()
            return Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        OpenAI = _load_openai_client()
        if model_name == "solar":
            return OpenAI(
                api_key=os.getenv("UPSTAGE_API_KEY"),
                base_url=self.model_info[model_name]["base_url"],
            )
        if model_name in ["local1", "local2"]:
            return OpenAI(api_key="EMPTY", base_url=self.model_info[model_name]["base_url"])
        if model_name == "solar_strawberry":
            return OpenAI(
                api_key=os.getenv("SOLAR_STRAWBERRY_API_KEY"),
                base_url=self.model_info[model_name]["base_url"],
            )
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def set_version(self, version: str):
        self.model_id = version
        return self

    def set_temperature(self, temperature: float):
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        self.temperature = temperature
        return self

    def set_top_p(self, top_p: float):
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("Top_p must be between 0.0 and 1.0")
        self.top_p = top_p
        return self

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        prices = self.token_prices.get(
            self.name, {"input_price_per_1k": 0, "output_price_per_1k": 0}
        )
        input_cost = (input_tokens / 1000) * prices["input_price_per_1k"]
        output_cost = (output_tokens / 1000) * prices["output_price_per_1k"]
        return input_cost + output_cost

    def _new_metadata(self) -> Dict[str, Any]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "duration": 0.0,
            "model": self.model_id,
        }

    def _chat_params(self, messages: list[dict[str, str]]) -> Dict[str, Any]:
        params: Dict[str, Any] = {"model": self.model_id, "messages": messages}
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if "solar" in self.model_id.lower():
            params["stream"] = False
        return params

    def _apply_openai_usage(self, metadata: Dict[str, Any], usage: Any) -> None:
        if not usage:
            return
        metadata["input_tokens"] = getattr(usage, "prompt_tokens", 0)
        metadata["output_tokens"] = getattr(usage, "completion_tokens", 0)
        metadata["total_tokens"] = getattr(
            usage,
            "total_tokens",
            metadata["input_tokens"] + metadata["output_tokens"],
        )
        metadata["cost"] = self._calculate_cost(
            metadata["input_tokens"], metadata["output_tokens"]
        )

    def _apply_anthropic_usage(self, metadata: Dict[str, Any], usage: Any) -> None:
        metadata["input_tokens"] = getattr(usage, "input_tokens", 0)
        metadata["output_tokens"] = getattr(usage, "output_tokens", 0)
        metadata["total_tokens"] = metadata["input_tokens"] + metadata["output_tokens"]
        metadata["cost"] = self._calculate_cost(
            metadata["input_tokens"], metadata["output_tokens"]
        )

    def _create_handler(self):
        def handler(question, system_prompt=None, user_prompt=None) -> Tuple[str, Dict[str, Any]]:
            start_time = time.time()
            metadata = self._new_metadata()

            try:
                system_prompt = system_prompt or self.system_prompt or ""
                user_prompt = user_prompt or self.user_prompt or ""
                messages = create_messages(question, system_prompt, user_prompt)

                if "claude" in self.model_id.lower():
                    response = self.client.messages.create(
                        model=self.model_id,
                        max_tokens=4096,
                        system=messages[0]["content"],
                        messages=[messages[1]],
                    )
                    self._apply_anthropic_usage(metadata, response.usage)
                    return response.content[0].text, metadata

                response = self.client.chat.completions.create(**self._chat_params(messages))
                self._apply_openai_usage(metadata, getattr(response, "usage", None))
                return response.choices[0].message.content, metadata
            except Exception as exc:
                return f"Error: {exc}", metadata
            finally:
                metadata["duration"] = time.time() - start_time

        return handler

    def ask(self, question, system_prompt=None, user_prompt=None) -> Tuple[str, Dict[str, Any]]:
        return self.handler(question, system_prompt, user_prompt)

    def ask_simple(self, question, system_prompt=None, user_prompt=None) -> str:
        answer, _ = self.ask(question, system_prompt, user_prompt)
        return answer
