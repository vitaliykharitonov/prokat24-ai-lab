from typing import Protocol

from prokat24_ai_lab.evals.semantic_grading import (
    Judge,
    JudgeInput,
    JudgeResult,
    Rubric,
    build_judge_prompt,
)

OPENAI_JUDGE_MODEL = "gpt-5.4-mini-2026-03-17"


class OutputTextProtocol(Protocol):
    type: str
    parsed: JudgeResult | None


class MessageProtocol(Protocol):
    type: str
    content: list[OutputTextProtocol]


class ResponseProtocol(Protocol):
    output: list[MessageProtocol]


class ResponsesProtocol(Protocol):
    def parse(
        self,
        *,
        text_format: type[JudgeResult],
        model: str,
        input: str,
    ) -> ResponseProtocol: ...


class JudgeClientProtocol(Protocol):
    responses: ResponsesProtocol


def create_openai_judge(
    client: JudgeClientProtocol,
    model: str = OPENAI_JUDGE_MODEL,
) -> Judge:
    def judge(judge_input: JudgeInput, rubric: Rubric) -> JudgeResult:
        prompt = build_judge_prompt(judge_input, rubric)
        response = client.responses.parse(text_format=JudgeResult, model=model, input=prompt)

        message = response.output[0]

        if message.type != "message":
            raise ValueError("Wrong message type")

        output_text = message.content[0]

        if output_text.type != "output_text":
            raise ValueError("Wrong output text type")

        parsed_result = output_text.parsed

        if parsed_result is None:
            raise ValueError("Wrong parsed result")

        return parsed_result

    return judge
