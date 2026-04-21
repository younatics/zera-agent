import unittest

from agent.common.api_client import Model, create_messages


class FakeUsage:
    prompt_tokens = 120
    completion_tokens = 30
    total_tokens = 150


class FakeMessage:
    content = "fake answer"


class FakeChoice:
    message = FakeMessage()


class FakeResponse:
    usage = FakeUsage()
    choices = [FakeChoice()]


class FakeCompletions:
    def __init__(self):
        self.last_params = None

    def create(self, **params):
        self.last_params = params
        return FakeResponse()


class FakeChat:
    def __init__(self):
        self.completions = FakeCompletions()


class FakeOpenAIClient:
    def __init__(self):
        self.chat = FakeChat()


class ApiClientTest(unittest.TestCase):
    def test_create_messages_combines_user_prompt_and_question(self):
        messages = create_messages("What is 2+2?", "system", "answer briefly")

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "system"},
                {"role": "user", "content": "answer briefly\n\nWhat is 2+2?"},
            ],
        )

    def test_invalid_model_name_raises_before_loading_sdk(self):
        with self.assertRaises(ValueError):
            Model("missing-model", client=FakeOpenAIClient())

    def test_openai_compatible_model_uses_injected_client(self):
        client = FakeOpenAIClient()
        model = Model("solar", client=client).set_temperature(0.2).set_top_p(0.8)

        answer, metadata = model.ask("question", system_prompt="system", user_prompt="user")

        self.assertEqual(answer, "fake answer")
        self.assertEqual(metadata["input_tokens"], 120)
        self.assertEqual(metadata["output_tokens"], 30)
        self.assertEqual(metadata["total_tokens"], 150)
        self.assertAlmostEqual(metadata["cost"], (120 / 1000) * 0.0002 + (30 / 1000) * 0.0004)

        params = client.chat.completions.last_params
        self.assertEqual(params["model"], "solar-pro")
        self.assertEqual(params["temperature"], 0.2)
        self.assertEqual(params["top_p"], 0.8)
        self.assertFalse(params["stream"])

    def test_sampling_parameters_validate_range(self):
        model = Model("gpt4o", client=FakeOpenAIClient())

        with self.assertRaises(ValueError):
            model.set_temperature(1.1)
        with self.assertRaises(ValueError):
            model.set_top_p(-0.1)


if __name__ == "__main__":
    unittest.main()
