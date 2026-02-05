import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

from anthropic import Anthropic
from openai import OpenAI


def _compose_user_content(question: str, user_prompt: str) -> str:
    """Join user prompt and question with clean separation."""

    segments = [segment for segment in (user_prompt.strip(), question.strip()) if segment]
    return "\n\n".join(segments)


def create_messages(question: str, system_prompt: str, user_prompt: str) -> list[Dict[str, str]]:
    """Create OpenAI style chat messages."""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": _compose_user_content(question, user_prompt)},
    ]


@dataclass(frozen=True)
class _ModelConfig:
    """Configuration describing how to initialise a model client."""

    name: str
    description: str
    default_version: str
    provider: str  # "openai" or "anthropic"
    base_url: Optional[str] = None
    api_key_env: Optional[str] = None
    default_api_key: Optional[str] = None


class Model:
    """Convenience wrapper around third-party LLM clients."""

    # Model-specific pricing information (USD per input token, per output token)
    token_prices: Dict[str, Dict[str, float]] = {
        "solar": {"input_price_per_1k": 0.0002, "output_price_per_1k": 0.0004},
        "gpt4o": {"input_price_per_1k": 0.005, "output_price_per_1k": 0.015},
        "claude": {"input_price_per_1k": 0.003, "output_price_per_1k": 0.015},
        "local1": {"input_price_per_1k": 0.0, "output_price_per_1k": 0.0},
        "local2": {"input_price_per_1k": 0.0, "output_price_per_1k": 0.0},
        "solar_strawberry": {"input_price_per_1k": 0.0002, "output_price_per_1k": 0.0004},
    }

    model_configs: Dict[str, _ModelConfig] = {
        "solar": _ModelConfig(
            name="Solar",
            description="Upstage's Solar model",
            default_version="solar-pro",
            provider="openai",
            base_url="https://api.upstage.ai/v1",
            api_key_env="SOLAR_API_KEY",
        ),
        "gpt4o": _ModelConfig(
            name="GPT-4",
            description="OpenAI's GPT-4 model",
            default_version="gpt-4o",
            provider="openai",
            api_key_env="OPENAI_API_KEY",
        ),
        "claude": _ModelConfig(
            name="Claude",
            description="Anthropic's Claude 3.5 Sonnet",
            default_version="claude-3-5-sonnet-20240620",
            provider="anthropic",
            api_key_env="ANTHROPIC_API_KEY",
        ),
        "local1": _ModelConfig(
            name="Local Model 1",
            description="First Mistral model running on local server",
            default_version="/data/project/private/kyle/hf_models/Meta-Llama-3-70B-Instruct",
            provider="openai",
            base_url="http://localhost:8001/v1",
            default_api_key="EMPTY",
        ),
        "local2": _ModelConfig(
            name="Local Model 2",
            description="Second Mistral model running on local server",
            default_version="/data/project/private/kyle/hf_models/Mistral-7B-Instruct-v0.3",
            provider="openai",
            base_url="http://localhost:8002/v1",
            default_api_key="EMPTY",
        ),
        "solar_strawberry": _ModelConfig(
            name="Solar-Strawberry",
            description="Upstage's Solar-Strawberry model",
            default_version="Solar-Strawberry",
            provider="openai",
            base_url="https://r-api.toy.x.upstage.ai/v1/",
            api_key_env="SOLAR_STRAWBERRY_API_KEY",
        ),
    }

    @classmethod
    def _config_to_dict(cls, config: _ModelConfig) -> Dict[str, Optional[str]]:
        return {
            "name": config.name,
            "description": config.description,
            "default_version": config.default_version,
            "base_url": config.base_url,
            "provider": config.provider,
            "api_key_env": config.api_key_env,
        }

    @classmethod
    def get_model_info(cls, model_name: str) -> Optional[Dict[str, Optional[str]]]:
        config = cls.model_configs.get(model_name)
        return cls._config_to_dict(config) if config else None

    @classmethod
    def get_all_model_info(cls) -> Dict[str, Dict[str, Optional[str]]]:
        return {name: cls._config_to_dict(config) for name, config in cls.model_configs.items()}

    def __init__(
        self,
        model_name: str,
        version: Optional[str] = None,
        *,
        client: Optional[Any] = None,
        client_factory: Optional[Callable[[str, _ModelConfig], Any]] = None,
    ) -> None:
        """Initialise a model wrapper.

        Args:
            model_name: Identifier from :pyattr:`model_configs`.
            version: Optional explicit model version; defaults to configuration value.
            client: Pre-configured client instance (primarily for testing).
            client_factory: Callable used to create a client when ``client`` is not provided.
        """

        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.model_configs.keys())}")

        self.name = model_name
        self.config = self.model_configs[model_name]
        self.model_id = version or self.config.default_version
        self.system_prompt: Optional[str] = None
        self.user_prompt: Optional[str] = None
        self.temperature: Optional[float] = None
        self.top_p: Optional[float] = None

        self.client = client or self._build_client(client_factory)

        self.handler = self._create_handler()

    def _build_client(self, client_factory: Optional[Callable[[str, _ModelConfig], Any]]) -> Any:
        if client_factory is not None:
            client = client_factory(self.name, self.config)
            if client is None:
                raise ValueError("client_factory must return a client instance")
            return client

        if self.config.provider == "anthropic":
            return Anthropic(api_key=self._resolve_api_key())

        if self.config.provider == "openai":
            kwargs: Dict[str, Any] = {}
            api_key = self._resolve_api_key()
            if api_key is not None:
                kwargs["api_key"] = api_key
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            return OpenAI(**kwargs)

        raise ValueError(f"Unsupported provider '{self.config.provider}' for model '{self.name}'")

    def _resolve_api_key(self) -> Optional[str]:
        if self.config.api_key_env:
            api_key = os.getenv(self.config.api_key_env)
            if not api_key:
                raise EnvironmentError(
                    f"Environment variable {self.config.api_key_env} is required for model '{self.name}'"
                )
            return api_key
        return self.config.default_api_key

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        input_tokens = getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", 0))
        output_tokens = getattr(usage, "completion_tokens", getattr(usage, "output_tokens", 0))
        total_tokens = getattr(usage, "total_tokens", input_tokens + output_tokens)
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }

    def _prepare_anthropic_payload(self, question: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        user_content = _compose_user_content(question, user_prompt)
        return {
            "model": self.model_id,
            "max_tokens": 4096,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_content},
                    ],
                }
            ],
        }

    def _extract_anthropic_text(self, response: Any) -> str:
        parts = []
        for block in getattr(response, "content", []):
            text = getattr(block, "text", None)
            if text is None and isinstance(block, dict):
                text = block.get("text")
            if text:
                parts.append(text)
        return "\n".join(parts)

    def set_version(self, version: str):
        """Set model version."""
        self.model_id = version
        return self

    def set_temperature(self, temperature: float):
        """Set temperature value. (0.0 ~ 1.0)"""
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        self.temperature = temperature
        return self

    def set_top_p(self, top_p: float):
        """Set top_p value. (0.0 ~ 1.0)"""
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("Top_p must be between 0.0 and 1.0")
        self.top_p = top_p
        return self

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage."""
        prices = self.token_prices.get(self.name, {"input_price_per_1k": 0, "output_price_per_1k": 0})
        input_cost = (input_tokens / 1000) * prices["input_price_per_1k"]
        output_cost = (output_tokens / 1000) * prices["output_price_per_1k"]
        return input_cost + output_cost

    def _create_handler(self):
        def handler(question, system_prompt=None, user_prompt=None) -> Tuple[str, Dict[str, Any]]:
            start_time = time.time()
            metadata = {
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "duration": 0.0,
                "model": self.model_id,
            }

            try:
                system_prompt = system_prompt or self.system_prompt or ""
                user_prompt = user_prompt or self.user_prompt or ""

                if self.config.provider == "anthropic":
                    payload = self._prepare_anthropic_payload(question, system_prompt, user_prompt)
                    response = self.client.messages.create(**payload)
                    answer = self._extract_anthropic_text(response)
                else:
                    messages = create_messages(question, system_prompt, user_prompt)
                    api_params = {
                        "model": self.model_id,
                        "messages": messages,
                    }

                    if self.temperature is not None:
                        api_params["temperature"] = self.temperature

                    if self.top_p is not None:
                        api_params["top_p"] = self.top_p

                    if "solar" in self.model_id:
                        api_params["stream"] = False

                    response = self.client.chat.completions.create(**api_params)
                    answer = response.choices[0].message.content

                usage = self._extract_usage(response)
                metadata.update(usage)
                metadata["cost"] = self._calculate_cost(metadata["input_tokens"], metadata["output_tokens"])
                metadata["duration"] = time.time() - start_time

                return answer, metadata

            except Exception as e:
                metadata["duration"] = time.time() - start_time
                metadata["error"] = str(e)
                return f"Error: {e}", metadata

        return handler

    def ask(self, question, system_prompt=None, user_prompt=None) -> Tuple[str, Dict[str, Any]]:
        """Ask the model a question and return the response and metadata."""
        answer, metadata = self.handler(question, system_prompt, user_prompt)
        print("Done.")
        return answer, metadata

    def ask_simple(self, question, system_prompt=None, user_prompt=None) -> str:
        """Method that returns only simple response for backward compatibility"""
        answer, _ = self.ask(question, system_prompt, user_prompt)
        return answer

    @classmethod
    def get_available_models(cls):
        return list(cls.model_configs.keys())


# Usage example:
# model = Model("gpt4o")
# answer = model.ask(input_prompt)
