from typing import Any

import pytest

from prokat24_ai_lab.evals.semantic_grading import (
    JudgeInput,
    JudgeResult,
    Rubric,
    build_judge_prompt,
)
from prokat24_ai_lab.judges.openai_judge import create_openai_judge


class FakeResponses:
    def __init__(
        self,
        parsed: JudgeResult | None,
        message_type: str = "message",
        output_text_type: str = "output_text",
    ) -> None:
        self.parsed = parsed
        self.message_type = message_type
        self.output_text_type = output_text_type

    def parse(
        self,
        *,
        text_format: type[JudgeResult],
        model: str,
        input: str,
    ) -> Any:
        self.text_format = text_format
        self.model = model
        self.input = input
        return FakeResponse(self.parsed, self.message_type, self.output_text_type)


class FakeClient:
    def __init__(
        self,
        parsed: JudgeResult | None,
        message_type: str = "message",
        output_text_type: str = "output_text",
    ) -> None:
        self.responses = FakeResponses(parsed, message_type, output_text_type)


class FakeOutputText:
    def __init__(self, parsed: JudgeResult | None, output_text_type: str = "output_text") -> None:
        self.type = output_text_type
        self.parsed = parsed


class FakeMessage:
    def __init__(
        self,
        parsed: JudgeResult | None,
        message_type: str = "message",
        output_text_type: str = "output_text",
    ) -> None:
        self.type = message_type
        self.content = [FakeOutputText(parsed, output_text_type)]


class FakeResponse:
    def __init__(
        self,
        parsed: JudgeResult | None,
        message_type: str = "message",
        output_text_type: str = "output_text",
    ) -> None:
        self.output = [FakeMessage(parsed, message_type, output_text_type)]


def test_create_openai_judge_calls_responses_api() -> None:
    expected_result = JudgeResult(
        score=5,
        passed=True,
        reason="The answer is correct.",
        failed_criteria=[],
    )
    client = FakeClient(expected_result)

    judge = create_openai_judge(client, model="test-model")

    judge_input: JudgeInput = {
        "question": "Чи безкоштовна доставка?",
        "context": "Доставка по Києву — від 200 грн.",
        "answer": "Доставка коштує від 200 грн.",
    }
    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must correctly state the delivery price.",
            }
        ],
    }

    expected_prompt = build_judge_prompt(judge_input, rubric)

    result = judge(judge_input, rubric)

    assert result == expected_result
    assert client.responses.text_format is JudgeResult
    assert client.responses.model == "test-model"
    assert client.responses.input == expected_prompt


def test_create_openai_judge_raises_when_parsed_result_is_none() -> None:
    client = FakeClient(None)
    judge = create_openai_judge(client, model="test-model")

    judge_input: JudgeInput = {
        "question": "Чи безкоштовна доставка?",
        "context": "Доставка по Києву — від 200 грн.",
        "answer": "Доставка коштує від 200 грн.",
    }

    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must correctly state the delivery price.",
            }
        ],
    }

    with pytest.raises(ValueError, match="Wrong parsed result"):
        judge(judge_input, rubric)


def test_create_openai_judge_raises_when_message_type_is_wrong() -> None:
    expected_result = JudgeResult(
        score=5,
        passed=True,
        reason="The answer is correct.",
        failed_criteria=[],
    )
    client = FakeClient(expected_result, message_type="wrong_type")

    judge = create_openai_judge(client, model="test-model")

    judge_input: JudgeInput = {
        "question": "Чи безкоштовна доставка?",
        "context": "Доставка по Києву — від 200 грн.",
        "answer": "Доставка коштує від 200 грн.",
    }
    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must correctly state the delivery price.",
            }
        ],
    }

    with pytest.raises(ValueError, match="Wrong message type"):
        judge(judge_input, rubric)


def test_create_openai_judge_raises_when_output_text_type_is_wrong() -> None:
    expected_result = JudgeResult(
        score=5,
        passed=True,
        reason="The answer is correct.",
        failed_criteria=[],
    )
    client = FakeClient(expected_result, output_text_type="wrong_type")

    judge = create_openai_judge(client, model="test-model")

    judge_input: JudgeInput = {
        "question": "Чи безкоштовна доставка?",
        "context": "Доставка по Києву — від 200 грн.",
        "answer": "Доставка коштує від 200 грн.",
    }
    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must correctly state the delivery price.",
            }
        ],
    }

    with pytest.raises(ValueError, match="Wrong output text type"):
        judge(judge_input, rubric)
