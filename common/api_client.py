import base64
import json
import os
from openai import OpenAI
from anthropic import Anthropic
from typing import Optional, Dict, Any


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
        }
    }

    @classmethod
    def get_model_info(cls, model_name):
        return cls.model_info.get(model_name)

    @classmethod
    def get_all_model_info(cls):
        return cls.model_info

    def __init__(self, model_name: str):
        """모델 초기화"""
        if model_name not in self.model_info:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.model_info.keys())}")
        
        self.name = model_name
        self.model_id = self.model_info[model_name]["default_version"]
        self.system_prompt = None
        self.user_prompt = None
        
        # Initialize appropriate client
        if model_name == "claude":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif model_name == "solar":
            self.client = OpenAI(
                api_key=os.getenv("SOLAR_API_KEY"),
                base_url=self.model_info[model_name]["base_url"]
            )
        else:  # gpt4o
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.handler = self._create_handler()

    def set_version(self, version: str):
        """모델 버전을 설정합니다."""
        self.model_id = version
        return self

    def _create_handler(self):
        def handler(question, system_prompt=None, user_prompt=None):
            try:
                system_prompt = system_prompt or self.system_prompt or "You are a helpful assistant."
                user_prompt = user_prompt or self.user_prompt or "Hello! I'm here to help you. Please let me know what you need assistance with, and I'll do my best to provide clear and helpful responses. Feel free to ask me anything!"
                messages = create_messages(question, system_prompt, user_prompt)
                if "claude" in self.model_id:
                    response = self.client.messages.create(
                        model=self.model_id,
                        max_tokens=1000,
                        system=messages[0]["content"],
                        messages=[messages[1]]
                    )
                    return response.content[0].text
                else:
                    response = self.client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        stream=False if "solar" in self.model_id else None,
                    )
                    return response.choices[0].message.content
            except Exception as e:
                return f"Error: {e}"
        return handler

    def ask(self, question, system_prompt=None, user_prompt=None):
        answer = self.handler(question, system_prompt, user_prompt)
        print("Done.")
        return answer

    @classmethod
    def get_available_models(cls):
        return list(cls.model_info.keys())


# 사용 예시:
# model = Model("gpt4o")
# answer = model.ask(input_prompt)
