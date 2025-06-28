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
    # 모델별 가격 정보 (입력 토큰당, 출력 토큰당 USD)
    token_prices = {
        "solar": {
            "input_price_per_1k": 0.0002,  # 예시 가격
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
            "input_price_per_1k": 0.0,  # 로컬 모델은 무료
            "output_price_per_1k": 0.0
        },
        "local2": {
            "input_price_per_1k": 0.0,  # 로컬 모델은 무료
            "output_price_per_1k": 0.0
        },
        "solar_strawberry": {
            "input_price_per_1k": 0.0002,  # 예시 가격
            "output_price_per_1k": 0.0004
        }
    }

    model_info = {
        "solar": {
            "name": "Solar",
            "description": "Upstage의 Solar 모델",
            "default_version": "solar-pro",
            "base_url": "https://api.upstage.ai/v1"
        },
        "gpt4o": {
            "name": "GPT-4",
            "description": "OpenAI의 GPT-4 모델",
            "default_version": "gpt-4o",
            "base_url": None
        },
        "claude": {
            "name": "Claude",
            "description": "Anthropic의 Claude 3.5 Sonnet",
            "default_version": "claude-3-5-sonnet-20240620",
            "base_url": None
        },
        "local1": {
            "name": "Local Model 1",
            "description": "로컬 서버에서 실행되는 첫 번째 미스트랄 모델",
            "default_version": "/data/project/private/kyle/hf_models/Meta-Llama-3-70B-Instruct",
            "base_url": "http://localhost:8001/v1"
        },
        "local2": {
            "name": "Local Model 2",
            "description": "로컬 서버에서 실행되는 두 번째 미스트랄 모델",
            "default_version": "/data/project/private/kyle/hf_models/Mistral-7B-Instruct-v0.3",
            "base_url": "http://localhost:8002/v1"
        },
        "solar_strawberry": {
            "name": "Solar-Strawberry",
            "description": "Upstage의 Solar-Strawberry 모델",
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
        """모델 초기화"""
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
        """모델 버전을 설정합니다."""
        self.model_id = version
        return self

    def set_temperature(self, temperature: float):
        """temperature 값을 설정합니다. (0.0 ~ 1.0)"""
        if not 0.0 <= temperature <= 1.0:
            raise ValueError("Temperature must be between 0.0 and 1.0")
        self.temperature = temperature
        return self

    def set_top_p(self, top_p: float):
        """top_p 값을 설정합니다. (0.0 ~ 1.0)"""
        if not 0.0 <= top_p <= 1.0:
            raise ValueError("Top_p must be between 0.0 and 1.0")
        self.top_p = top_p
        return self

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """토큰 사용량을 기반으로 비용을 계산합니다."""
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
                    
                    # Claude API에서 토큰 사용량 정보 추출
                    metadata["input_tokens"] = response.usage.input_tokens
                    metadata["output_tokens"] = response.usage.output_tokens
                    metadata["total_tokens"] = metadata["input_tokens"] + metadata["output_tokens"]
                    metadata["cost"] = self._calculate_cost(metadata["input_tokens"], metadata["output_tokens"])
                    metadata["duration"] = time.time() - start_time
                    
                    return response.content[0].text, metadata
                else:
                    # API 파라미터 준비
                    api_params = {
                        "model": self.model_id,
                        "messages": messages,
                    }
                    
                    # temperature가 설정된 경우에만 추가
                    if self.temperature is not None:
                        api_params["temperature"] = self.temperature
                    
                    # top_p가 설정된 경우에만 추가
                    if self.top_p is not None:
                        api_params["top_p"] = self.top_p
                    
                    # Solar 모델의 경우 stream=False 설정
                    if "solar" in self.model_id:
                        api_params["stream"] = False
                    
                    response = self.client.chat.completions.create(**api_params)
                    
                    # OpenAI 호환 API에서 토큰 사용량 정보 추출
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
        """모델에게 질문하고 응답과 메타데이터를 반환합니다."""
        answer, metadata = self.handler(question, system_prompt, user_prompt)
        print("Done.")
        return answer, metadata

    def ask_simple(self, question, system_prompt=None, user_prompt=None) -> str:
        """기존 호환성을 위한 간단한 응답만 반환하는 메서드"""
        answer, _ = self.ask(question, system_prompt, user_prompt)
        return answer

    @classmethod
    def get_available_models(cls):
        return list(cls.model_info.keys())


# 사용 예시:
# model = Model("gpt4o")
# answer = model.ask(input_prompt)
