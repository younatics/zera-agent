import base64
import json
import os
from openai import OpenAI


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
    _models = {
        "gpt4o": "gpt-4o",
        "claude": "claude-3-sonnet-20240229",
        "solar": "solar-pro"
    }

    _base_urls = {
        "solar": "https://api.upstage.ai/v1"
    }

    def __init__(self, model_name):
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self._models.keys())}")
        
        self.name = model_name
        self.model_id = self._models[model_name]
        self.handler = self._create_handler()

    def _create_handler(self):
        def handler(client, prompt, system_message="You are a helpful assistant."):
            try:
                messages = create_messages(prompt, system_message)
                if "claude" in self.model_id:
                    response = client.messages.create(
                        model=self.model_id,
                        max_tokens=1000,
                        system=messages[0]["content"],
                        messages=[messages[1]]
                    )
                    return response.content[0].text
                else:
                    # Solar Pro의 경우 base_url 설정이 필요
                    if "solar" in self.model_id:
                        client = OpenAI(
                            api_key=os.getenv("SOLAR_API_KEY"),
                            base_url=self._base_urls["solar"]
                        )
                    
                    response = client.chat.completions.create(
                        model=self.model_id,
                        messages=messages,
                        stream=False if "solar" in self.model_id else None,
                    )
                    return response.choices[0].message.content
            except Exception as e:
                return f"Error: {e}"
        return handler

    def ask(self, client, prompt, system_message="You are a helpful assistant."):
        answer = self.handler(client, prompt, system_message)
        print("Done.")
        return answer

    @classmethod
    def get_available_models(cls):
        return list(cls._models.keys())


# 사용 예시:
# model = Model("gpt4o")
# answer = model.ask(client, input_prompt)
