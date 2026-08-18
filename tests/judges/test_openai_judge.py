from typing import Any

import pytest

from prokat24_ai_lab.evals.semantic_grading import (
    CriterionJudgeResult,
    CriterionResult,
    JudgeInput,
    JudgeResult,
    Rubric,
    build_judge_prompt,
)
from prokat24_ai_lab.judges.openai_judge import create_openai_criterion_judge, create_openai_judge


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


def test_create_openai_criterion_judge_returns_criterion_result() -> None:
    judge_input: JudgeInput = {
        "question": "Скільки коштує доставка по Києву?",
        "context": (
            "Доставка по Києву коштує від 200 грн. Пункт прокату працює щодня з 10:00 до 19:00."
        ),
        "answer": (
            "Доставка по Києву коштує від 200 грн. Пункт прокату працює щодня з 10:00 до 19:00."
        ),
    }

    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must correctly state the delivery price in Kyiv.",
            },
            {
                "name": "relevance",
                "description": "The answer must contain only information "
                "needed to answer the question.",
            },
            {
                "name": "groundedness",
                "description": "The answer must use only information from the provided context.",
            },
        ],
    }

    expected_result = CriterionJudgeResult(
        criteria=[
            CriterionResult(
                name="correctness",
                passed=True,
                reason="The delivery price is correct.",
            ),
            CriterionResult(
                name="relevance",
                passed=False,
                reason="The opening hours are not needed to answer the question.",
            ),
            CriterionResult(
                name="groundedness",
                passed=True,
                reason="All information is supported by the context.",
            ),
        ]
    )

    client = FakeClient(expected_result, output_text_type="output_text")

    judge = create_openai_criterion_judge(client, model="test-model")

    result = judge(judge_input, rubric)

    assert result == expected_result

    assert len(result.criteria) == 3

    assert result.criteria[0].name == "correctness"
    assert result.criteria[0].passed is True

    assert result.criteria[1].name == "relevance"
    assert result.criteria[1].passed is False

    assert result.criteria[2].name == "groundedness"
    assert result.criteria[2].passed is True

    assert client.responses.text_format is CriterionJudgeResult
    assert client.responses.model == "test-model"

    assert client.responses.input is not None
    assert "Return one result for every rubric criterion." in client.responses.input
    assert "correctness" in client.responses.input
    assert "relevance" in client.responses.input
    assert "groundedness" in client.responses.input


def test_create_openai_criterion_judge_raises_when_criterion_is_missing() -> None:
    judge_input: JudgeInput = {
        "question": "Скільки коштує доставка по Києву?",
        "context": "Доставка по Києву коштує від 200 грн.",
        "answer": "Доставка по Києву коштує від 200 грн.",
    }

    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must correctly state the delivery price in Kyiv.",
            },
            {
                "name": "relevance",
                "description": "The answer must contain only relevant information.",
            },
            {
                "name": "groundedness",
                "description": "The answer must use only information from the context.",
            },
        ],
    }

    parsed_result = CriterionJudgeResult(
        criteria=[
            CriterionResult(
                name="correctness",
                passed=True,
                reason="The delivery price is correct.",
            ),
            CriterionResult(
                name="groundedness",
                passed=True,
                reason="The answer is supported by the context.",
            ),
        ]
    )

    client = FakeClient(parsed_result)
    judge = create_openai_criterion_judge(client)

    with pytest.raises(ValueError):
        judge(judge_input, rubric)


def test_create_openai_criterion_judge_raises_on_wrong_message_type() -> None:
    judge_input: JudgeInput = {
        "question": "Скільки коштує доставка по Києву?",
        "context": "Доставка по Києву коштує від 200 грн.",
        "answer": "Доставка по Києву коштує від 200 грн.",
    }

    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must correctly state the delivery price in Kyiv.",
            }
        ],
    }

    parsed_result = CriterionJudgeResult(
        criteria=[
            CriterionResult(
                name="correctness",
                passed=True,
                reason="The delivery price is correct.",
            )
        ]
    )

    client = FakeClient(
        parsed_result,
        message_type="wrong_type",
    )
    judge = create_openai_criterion_judge(client)

    with pytest.raises(ValueError, match="Wrong message type"):
        judge(judge_input, rubric)


def test_create_openai_criterion_judge_raises_on_wrong_output_text_type() -> None:
    judge_input: JudgeInput = {
        "question": "Скільки коштує доставка по Києву?",
        "context": "Доставка по Києву коштує від 200 грн.",
        "answer": "Доставка по Києву коштує від 200 грн.",
    }

    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must correctly state the delivery price in Kyiv.",
            }
        ],
    }

    parsed_result = CriterionJudgeResult(
        criteria=[
            CriterionResult(
                name="correctness",
                passed=True,
                reason="The delivery price is correct.",
            )
        ]
    )

    client = FakeClient(
        parsed_result,
        output_text_type="wrong_type",
    )
    judge = create_openai_criterion_judge(client)

    with pytest.raises(ValueError, match="Wrong output text type"):
        judge(judge_input, rubric)


def test_create_openai_criterion_judge_raises_when_parsed_result_is_none() -> None:
    judge_input: JudgeInput = {
        "question": "Скільки коштує доставка по Києву?",
        "context": "Доставка по Києву коштує від 200 грн.",
        "answer": "Доставка по Києву коштує від 200 грн.",
    }

    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must correctly state the delivery price in Kyiv.",
            }
        ],
    }

    client = FakeClient(None)
    judge = create_openai_criterion_judge(client)

    with pytest.raises(ValueError, match="Wrong parsed result"):
        judge(judge_input, rubric)
