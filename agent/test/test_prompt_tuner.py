import unittest

from agent.core.prompt_tuner import PromptTuner


EMPTY_METADATA = {
    "input_tokens": 0,
    "output_tokens": 0,
    "total_tokens": 0,
    "cost": 0.0,
    "duration": 0.0,
}


class FakeModel:
    def __init__(self, name, responses):
        self.name = name
        self.responses = list(responses)
        self.calls = []

    def ask(self, question, system_prompt=None, user_prompt=None):
        self.calls.append(
            {
                "question": question,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        if not self.responses:
            return "", EMPTY_METADATA.copy()
        response = self.responses.pop(0)
        if isinstance(response, tuple):
            return response
        return response, EMPTY_METADATA.copy()


class FakeModelFactory:
    def __init__(self, responses_by_name=None):
        self.responses_by_name = responses_by_name or {}
        self.models = {}

    def __call__(self, name, version=None):
        model = FakeModel(name, self.responses_by_name.get(name, []))
        self.models[name] = model
        return model


def build_tuner(responses_by_name=None):
    factory = FakeModelFactory(responses_by_name)
    tuner = PromptTuner(
        model_name="generator",
        evaluator_model_name="evaluator",
        meta_prompt_model_name="meta",
        model_factory=factory,
    )
    return tuner, factory


class PromptTunerTest(unittest.TestCase):
    def test_parse_evaluation_response_scores_weighted_categories(self):
        tuner, _ = build_tuner()

        score, details = tuner._parse_evaluation_response(
            'prefix {"scores": {"correctness": {"score": "1", "weight": "0.7"}, '
            '"format": "PASS"}} suffix'
        )

        self.assertAlmostEqual(score, (1.0 * 0.7 + 0.5 * 0.5) / 1.2)
        self.assertEqual(details["category_scores"]["correctness"]["score"], 1.0)
        self.assertEqual(details["category_scores"]["format"]["current_state"], "PASS")

    def test_evaluate_output_updates_evaluator_stats(self):
        tuner, factory = build_tuner(
            {
                "evaluator": [
                    (
                        '{"scores": {"quality": {"score": "0.8", "weight": "0.5"}}}',
                        {
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                            "cost": 0.01,
                            "duration": 0.2,
                        },
                    )
                ]
            }
        )
        tuner.set_evaluation_prompt(
            system_prompt_template="{task_type}",
            user_prompt_template="{question}|{response}|{expected}|{task_description}",
        )

        score, details = tuner._evaluate_output(
            output="actual",
            expected="expected",
            question="question",
            task_type="task",
            task_description="description",
            iteration=2,
        )

        self.assertEqual(score, 0.8)
        self.assertIn("quality", details["category_scores"])
        self.assertEqual(tuner.evaluator_stats["total_calls"], 1)
        self.assertEqual(tuner.evaluator_stats["calls_by_iteration"][2]["calls"], 1)
        self.assertEqual(factory.models["evaluator"].calls[0]["system_prompt"], "task")

    def test_parse_meta_prompt_response_accepts_expected_sections(self):
        parsed = PromptTuner.parse_meta_prompt_response(
            "Task Type: Math\n"
            "Task Description: Solve arithmetic.\n"
            "System Prompt: You are careful.\n"
            "User Prompt: Show concise work."
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.task_type, "Math")
        self.assertEqual(parsed.task_description, "Solve arithmetic.")
        self.assertEqual(parsed.system_prompt, "You are careful.")
        self.assertEqual(parsed.user_prompt, "Show concise work.")

    def test_parse_meta_prompt_response_rejects_missing_sections(self):
        self.assertIsNone(PromptTuner.parse_meta_prompt_response("System Prompt: only one section"))

    def test_tune_prompt_runs_without_meta_prompt_or_external_services(self):
        tuner, factory = build_tuner(
            {
                "generator": ["model answer"],
                "evaluator": ['{"scores": {"correctness": {"score": 1, "weight": 1}}}'],
            }
        )

        results = tuner.tune_prompt(
            initial_system_prompt="system",
            initial_user_prompt="user",
            initial_test_cases=[{"question": "q1", "expected": "model answer"}],
            num_iterations=1,
            use_meta_prompt=False,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].avg_score, 1.0)
        self.assertEqual(results[0].best_avg_score, 1.0)
        self.assertEqual(tuner.model_stats["total_calls"], 1)
        self.assertEqual(len(factory.models["meta"].calls), 0)

    def test_save_results_to_csv_uses_standard_library(self):
        tuner, _ = build_tuner(
            {
                "generator": ["model answer"],
                "evaluator": ['{"scores": {"correctness": {"score": 1, "weight": 1}}}'],
            }
        )
        tuner.tune_prompt(
            initial_system_prompt="system",
            initial_user_prompt="user",
            initial_test_cases=[{"question": "q1", "expected": "model answer"}],
            num_iterations=1,
            use_meta_prompt=False,
        )

        csv_output = tuner.save_results_to_csv()

        self.assertIn("Iteration,Average Score", csv_output)
        self.assertIn("model answer", csv_output)


if __name__ == "__main__":
    unittest.main()
