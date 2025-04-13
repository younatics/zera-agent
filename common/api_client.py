import base64
import json
import os
from openai import OpenAI
from anthropic import Anthropic


def create_messages(prompt, system_message="You are a helpful assistant."):
    return [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": prompt
        }
    ]


class Model:
    models = {
        "gpt4o": "gpt-4o",
        "claude": "claude-3-sonnet-20240229",
        "solar": "solar-pro"
    }

    base_urls = {
        "solar": "https://api.upstage.ai/v1"
    }

    def __init__(self, model_name, system_prompt=None):
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.models.keys())}")
        
        self.name = model_name
        self.model_id = self.models[model_name]
        self.system_prompt = system_prompt
        
        # Initialize appropriate client
        if model_name == "claude":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif model_name == "solar":
            self.client = OpenAI(
                api_key=os.getenv("SOLAR_API_KEY"),
                base_url=self.base_urls["solar"]
            )
        else:  # gpt4o
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        self.handler = self._create_handler()

    def _create_handler(self):
        def handler(prompt, system_prompt=None):
            try:
                messages = create_messages(prompt, system_prompt or self.system_prompt or "You are a helpful assistant.")
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

    def ask(self, prompt, system_prompt=None):
        answer = self.handler(prompt, system_prompt)
        print("Done.")
        return answer

    @classmethod
    def get_available_models(cls):
        return list(cls.models.keys())


# 사용 예시:
# model = Model("gpt4o")
# answer = model.ask(input_prompt)
