from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple


def _get_value(item: Any, *keys: str, default: Any = "") -> Any:
    for key in keys:
        try:
            value = item[key]
        except (KeyError, TypeError):
            value = getattr(item, key, None)
        if value is not None:
            return value
    return default


def _records(data: Any) -> Iterable[Any]:
    if hasattr(data, "iterrows"):
        for _, row in data.iterrows():
            yield row
        return
    yield from data


def _validate_csv_columns(data: Any) -> None:
    required_columns = {"question", "expected_answer"}
    if hasattr(data, "columns"):
        missing_columns = required_columns - set(data.columns.tolist())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"CSV file requires the following columns: {missing}")


def _answer_to_choice(answer: Any) -> Any:
    if isinstance(answer, int):
        return chr(65 + answer)
    if isinstance(answer, str) and len(answer) == 1 and answer.isalpha():
        return answer
    return answer


def _display_row(question: Any, expected: Any) -> Dict[str, Any]:
    return {"question": question, "expected_answer": expected}


def build_test_cases(
    data: Any,
    dataset_type: str,
    max_display: int = 2000,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Convert app dataset rows into PromptTuner test cases and preview rows."""
    if dataset_type == "CSV":
        _validate_csv_columns(data)

    test_cases: List[Dict[str, Any]] = []
    display_data: List[Dict[str, Any]] = []

    for item in _records(data):
        if dataset_type == "Samsum":
            question = _get_value(item, "dialogue")
            expected = _get_value(item, "summary")
        elif dataset_type == "MeetingBank":
            question = _get_value(item, "transcript")
            expected = _get_value(item, "summary")
        elif dataset_type == "BBH":
            question = _get_value(item, "input", "question")
            expected = _get_value(item, "target", "answer")
        elif dataset_type == "MBPP":
            question = _get_value(item, "text", "prompt")
            expected = _get_value(item, "code")
        elif dataset_type == "XSum":
            question = _get_value(item, "document")
            expected = _get_value(item, "summary")
        elif dataset_type in ["MMLU", "MMLU Pro"]:
            choices = _get_value(item, "choices", default=[])
            choices_str = "\n".join(
                f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices)
            )
            question = f"{_get_value(item, 'question')}\n\nChoices:\n{choices_str}"
            expected = _answer_to_choice(_get_value(item, "answer"))
        elif dataset_type == "CNN":
            question = _get_value(item, "input", "article")
            expected = _normalize_summary(_get_value(item, "expected_answer", "summary"))
        elif dataset_type == "GSM8K":
            question = _get_value(item, "question")
            expected = _get_value(item, "answer")
        elif dataset_type == "TruthfulQA":
            question = _get_value(item, "input", "question")
            expected = _get_value(item, "target", "best_answer", "answer")
        elif dataset_type == "HellaSwag":
            choices = _get_value(item, "choices", default=[])
            choices_str = "\n".join(
                f"{chr(65 + index)}. {choice}" for index, choice in enumerate(choices)
            )
            question = (
                f"Activity: {_get_value(item, 'activity_label')}\n"
                f"Context: {_get_value(item, 'context')}\n\n"
                f"Complete the context with the most appropriate ending:\n{choices_str}"
            )
            expected = _answer_to_choice(_get_value(item, "answer"))
        elif dataset_type == "HumanEval":
            question = _get_value(item, "prompt")
            expected = _get_value(item, "canonical_solution")
        elif dataset_type == "CSV":
            question = _get_value(item, "question")
            expected = _get_value(item, "expected_answer")
        else:
            raise ValueError(f"Unsupported dataset type: {dataset_type}")

        test_cases.append({"question": question, "expected": expected})
        if len(display_data) < max_display:
            display_data.append(_display_row(question, expected))

    return test_cases, display_data


def _normalize_summary(summary: Any) -> str:
    if not isinstance(summary, str):
        return str(summary)
    return " ".join(
        line.strip()
        for line in summary.split("\n")
        if line.strip() and not line.strip().startswith(("-", "*"))
    )
