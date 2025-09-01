import base64
import json
import os
import time
from openai import OpenAI
from anthropic import Anthropic
from typing import Optional, Dict, Any, Tuple


def create_messages(question, system_prompt, user_prompt):
    messages = [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": f"{user_prompt}\n\n{question}"
        }
    ]
    
    return messages


class Model:
    # Model-specific pricing information (USD per input token, per output token)
    token_prices = {
        "solar": {
            "input_price_per_1k": 0.0002,  # Example price
            "output_price_per_1k": 0.0004
        },
        "gpt4o": {
            "input_price_per_1k": 0.005,
            "output_price_per_1k": 0.015
        },
        "claude": {
            "input_price_per_1k": 0.003,
            "output_price_per_1k": 0.015
        },
        "local1": {
            "input_price_per_1k": 0.0,  # Local models are free
            "output_price_per_1k": 0.0
        },
        "local2": {
            "input_price_per_1k": 0.0,  # Local models are free
            "output_price_per_1k": 0.0
        },
        "solar_strawberry": {
            "input_price_per_1k": 0.0002,  # Example price
            "output_price_per_1k": 0.0004
        }
    }

    model_info = {
        "solar": {
            "name": "Solar",
            "description": "Upstage's Solar model",
            "default_version": "solar-pro",
            "base_url": "https://api.upstage.ai/v1"
        },
        "gpt4o": {
            "name": "GPT-4",
            "description": "OpenAI's GPT-4 model",
            "default_version": "gpt-4o",
            "base_url": None
        },
        "claude": {
            "name": "Claude",
            "description": "Anthropic's Claude 3.5 Sonnet",
            "default_version": "claude-3-5-sonnet-20240620",
            "base_url": None
        },
        "local1": {
            "name": "Local Model 1",
            "description": "First Mistral model running on local server",
            "default_version": "/data/project/private/kyle/hf_models/Meta-Llama-3-70B-Instruct",
            "base_url": "http://localhost:8001/v1"
        },
        "local2": {
            "name": "Local Model 2",
            "description": "Second Mistral model running on local server",
            "default_version": "/data/project/private/kyle/hf_models/Mistral-7B-Instruct-v0.3",
            "base_url": "http://localhost:8002/v1"
        },
        "solar_strawberry": {
            "name": "Solar-Strawberry",
            "description": "Upstage's Solar-Strawberry model",
            "default_version": "Solar-Strawberry",
            "base_url": "https://r-api.toy.x.upstage.ai/v1/"
        },
    }

    @classmethod
    def get_model_info(cls, model_name):
        return cls.model_info.get(model_name)

    @classmethod
    def get_all_model_info(cls):
        return cls.model_info

    def __init__(self, model_name: str, version: Optional[str] = None):
        """Initialize model"""
        if model_name not in self.model_info:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.model_info.keys())}")
        
        self.name = model_name
        self.model_id = version or self.model_info[model_name]["default_version"]
        self.system_prompt = None
        self.user_prompt = None
        self.temperature = None
        self.top_p = None
        
        # Initialize appropriate client
        if model_name == "claude":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif model_name == "solar":
            self.client = OpenAI(
                api_key=os.getenv("SOLAR_API_KEY"),
                base_url=self.model_info[model_name]["base_url"]
            )
        elif model_name in ["local1", "local2"]:
            self.client = OpenAI(
                api_key="EMPTY",
                base_url=self.model_info[model_name]["base_url"]
            )
        elif model_name == "solar_strawberry":
            self.client = OpenAI(
                api_key=os.getenv("SOLAR_STRAWBERRY_API_KEY"),
                base_url=self.model_info[model_name]["base_url"]
            )
        else:  # gpt4o
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.handler = self._create_handler()

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
                "model": self.model_id
            }
            
            try:
                system_prompt = system_prompt or self.system_prompt or ""
                user_prompt = user_prompt or self.user_prompt or ""
                messages = create_messages(question, system_prompt, user_prompt)
                
                if "claude" in self.model_id:
                    response = self.client.messages.create(
                        model=self.model_id,
                        max_tokens=4096,
                        system=messages[0]["content"],
                        messages=[messages[1]]
                    )
                    
                    # Extract token usage information from Claude API
                    metadata["input_tokens"] = response.usage.input_tokens
                    metadata["output_tokens"] = response.usage.output_tokens
                    metadata["total_tokens"] = metadata["input_tokens"] + metadata["output_tokens"]
                    metadata["cost"] = self._calculate_cost(metadata["input_tokens"], metadata["output_tokens"])
                    metadata["duration"] = time.time() - start_time
                    
                    return response.content[0].text, metadata
                else:
                    # Prepare API parameters
                    api_params = {
                        "model": self.model_id,
                        "messages": messages,
                    }
                    
                    # Add only if temperature is set
                    if self.temperature is not None:
                        api_params["temperature"] = self.temperature
                    
                    # Add only if top_p is set
                    if self.top_p is not None:
                        api_params["top_p"] = self.top_p
                    
                    # Set stream=False for Solar models
                    if "solar" in self.model_id:
                        api_params["stream"] = False
                    
                    response = self.client.chat.completions.create(**api_params)
                    
                    # Extract token usage information from OpenAI-compatible API
                    if response.usage:
                        metadata["input_tokens"] = response.usage.prompt_tokens if hasattr(response.usage, 'prompt_tokens') else 0
                        metadata["output_tokens"] = response.usage.completion_tokens if hasattr(response.usage, 'completion_tokens') else 0
                        metadata["total_tokens"] = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else metadata["input_tokens"] + metadata["output_tokens"]
                    metadata["cost"] = self._calculate_cost(metadata["input_tokens"], metadata["output_tokens"])
                    metadata["duration"] = time.time() - start_time
                    
                    return response.choices[0].message.content, metadata
                    
            except Exception as e:
                metadata["duration"] = time.time() - start_time
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
        return list(cls.model_info.keys())


# Usage example:
# model = Model("gpt4o")
# answer = model.ask(input_prompt)
