from collections.abc import Callable
from typing import TypedDict

from pydantic import BaseModel, Field


class RubricCriterion(TypedDict):
    name: str
    description: str


class Rubric(TypedDict):
    name: str
    criteria: list[RubricCriterion]


class JudgeInput(TypedDict):
    question: str
    context: str
    answer: str


class JudgeResult(BaseModel):
    score: int = Field(ge=1, le=5)
    passed: bool
    reason: str
    failed_criteria: list[str]


Judge = Callable[[JudgeInput, Rubric], JudgeResult]

JUDGE_INSTRUCTIONS: str = (
    "Judge instructions:\n"
    "You are evaluating an assistant answer.\n"
    "Evaluate it strictly using the provided rubric.\n"
    "Use only the provided context.\n"
    "Do not use outside knowledge."
)

RETURN_INSTRUCTIONS: str = (
    "Return:\n"
    "- score: integer from 1 to 5\n"
    "- passed: true or false\n"
    "- reason: concise explanation\n"
    "- failed_criteria: names of failed rubric criteria"
)


def evaluate_with_judge(
    judge_input: JudgeInput,
    rubric: Rubric,
    judge: Judge,
) -> JudgeResult:
    return judge(judge_input, rubric)


def build_judge_prompt(
    judge_input: JudgeInput,
    rubric: Rubric,
) -> str:
    lines: list[str] = []
    lines.append(JUDGE_INSTRUCTIONS)
    lines.append("")
    lines.append("Rubric:")
    lines.append(rubric["name"])

    for criteria in rubric["criteria"]:
        lines.append(f"- {criteria['name']}: {criteria['description']}")

    lines.append("")
    lines.append("Question:")
    lines.append(judge_input["question"])
    lines.append("")
    lines.append("Context:")
    lines.append(judge_input["context"])
    lines.append("")
    lines.append("Assistant answer:")
    lines.append(judge_input["answer"])
    lines.append("")
    lines.append(RETURN_INSTRUCTIONS)

    return "\n".join(lines)
