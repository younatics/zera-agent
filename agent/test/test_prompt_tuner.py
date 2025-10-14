import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.core.prompt_tuner import PromptTuner


class StubModel:
    def __init__(self, responses):
        self._responses = iter(responses)
        self.calls = []

    def ask(self, question, system_prompt=None, user_prompt=None):
        self.calls.append((question, system_prompt, user_prompt))
        response = next(self._responses)
        metadata = {
            "input_tokens": 10,
            "output_tokens": 5,
            "total_tokens": 15,
            "cost": 0.0,
            "duration": 0.01,
            "model": "stub",
        }
        return response, metadata


class StubEvaluationModel(StubModel):
    pass


def test_prompt_tuner_with_stub_models(tmp_path):
    answer_model = StubModel(["Expected answer"])
    evaluation_payload = [
        '{"scores": {"quality": {"score": 1.0, "weight": 1.0, "current_state": "ok", "improvement_action": ""}}}'
    ]
    evaluator_model = StubEvaluationModel(evaluation_payload)
    meta_model = StubEvaluationModel([""])

    tuner = PromptTuner(
        model_name="solar",
        evaluator_model_name="solar",
        meta_prompt_model_name="solar",
        model=answer_model,
        evaluator=evaluator_model,
        meta_prompt_model=meta_model,
    )

    test_cases = [{"question": "What?", "expected": "Expected answer"}]

    results = tuner.tune_prompt(
        initial_system_prompt="System",
        initial_user_prompt="User",
        initial_test_cases=test_cases,
        num_iterations=1,
        use_meta_prompt=False,
    )

    assert len(results) == 1
    result = results[0]
    assert result.avg_score == 1.0
    assert result.test_case_results[0].actual_output == "Expected answer"
    assert answer_model.calls[0][0] == "What?"
    assert evaluator_model.calls
    assert meta_model.calls == []
