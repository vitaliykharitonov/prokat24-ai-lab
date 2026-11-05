from collections.abc import Sequence
from typing import Protocol, TypeVar, cast

from prokat24_ai_lab.evals.grading.semantic import (
    CriterionJudge,
    CriterionJudgeResult,
    Judge,
    JudgeInput,
    JudgeResult,
    Rubric,
    build_criterion_judge_prompt,
    build_judge_prompt,
    validate_criterion_judge_result,
)

OPENAI_JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"

JudgeResultT = TypeVar("JudgeResultT", JudgeResult, CriterionJudgeResult, covariant=True)


class OutputItemProtocol(Protocol):
    @property
    def type(self) -> str: ...


class ContentItemProtocol(Protocol):
    @property
    def type(self) -> str: ...


class OutputTextProtocol(ContentItemProtocol, Protocol[JudgeResultT]):
    @property
    def parsed(self) -> JudgeResultT | None: ...


class MessageProtocol(OutputItemProtocol, Protocol):
    @property
    def content(self) -> Sequence[ContentItemProtocol]: ...


class ResponseProtocol(Protocol):
    @property
    def output(self) -> Sequence[OutputItemProtocol]: ...


class ResponsesProtocol(Protocol):
    def parse(
        self,
        *,
        text_format: type[JudgeResultT],
        model: str,
        input: str,
    ) -> ResponseProtocol: ...


class JudgeClientProtocol(Protocol):
    @property
    def responses(self) -> ResponsesProtocol: ...


def create_openai_judge(
    client: JudgeClientProtocol,
    model: str = OPENAI_JUDGE_MODEL,
) -> Judge:
    def judge(judge_input: JudgeInput, rubric: Rubric) -> JudgeResult:
        prompt = build_judge_prompt(judge_input, rubric)
        response = client.responses.parse(text_format=JudgeResult, model=model, input=prompt)

        output_item = response.output[0]

        if output_item.type != "message":
            raise ValueError("Wrong message type")

        message = cast(MessageProtocol, output_item)

        content_item = message.content[0]

        if content_item.type != "output_text":
            raise ValueError("Wrong output text type")

        output_text = cast(OutputTextProtocol[JudgeResult], content_item)

        parsed_result = output_text.parsed

        if parsed_result is None:
            raise ValueError("Wrong parsed result")

        return parsed_result

    return judge


def create_openai_criterion_judge(
    client: JudgeClientProtocol,
    model: str = OPENAI_JUDGE_MODEL,
) -> CriterionJudge:
    def judge(judge_input: JudgeInput, rubric: Rubric) -> CriterionJudgeResult:
        prompt = build_criterion_judge_prompt(judge_input, rubric)
        response = client.responses.parse(
            text_format=CriterionJudgeResult,
            model=model,
            input=prompt,
        )

        output_item = response.output[0]

        if output_item.type != "message":
            raise ValueError("Wrong message type")

        message = cast(MessageProtocol, output_item)

        content_item = message.content[0]

        if content_item.type != "output_text":
            raise ValueError("Wrong output text type")

        output_text = cast(OutputTextProtocol[CriterionJudgeResult], content_item)

        parsed_result = output_text.parsed

        if parsed_result is None:
            raise ValueError("Wrong parsed result")

        validate_criterion_judge_result(parsed_result, rubric)

        return parsed_result

    return judge
