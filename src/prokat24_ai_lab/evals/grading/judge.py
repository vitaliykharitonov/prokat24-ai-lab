from collections.abc import Callable
from typing import Self, TypedDict

from pydantic import BaseModel, Field, model_validator


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

    @model_validator(mode="after")
    def validate_passed_with_failed_criteria(self) -> Self:
        if self.passed and self.failed_criteria:
            raise ValueError("Passed result cannot have failed criteria")

        return self


class CriterionResult(BaseModel):
    name: str
    passed: bool
    reason: str


class CriterionJudgeResult(BaseModel):
    criteria: list[CriterionResult]


class CriterionJudgeSummary(TypedDict):
    passed: bool
    failed_criteria: list[str]


Judge = Callable[[JudgeInput, Rubric], JudgeResult]

CriterionJudge = Callable[[JudgeInput, Rubric], CriterionJudgeResult]

JUDGE_INSTRUCTIONS: str = (
    "Judge instructions:\n"
    "You are evaluating an assistant answer.\n"
    "Evaluate it strictly using the provided rubric.\n"
    "Use only the provided context.\n"
    "Do not use outside knowledge.\n"
    "Evaluate every rubric criterion independently.\n"
    "Groundedness and relevance are separate criteria.\n"
    "Information can be fully supported by the context and still "
    "be irrelevant to the user's question.\n"
    "If the rubric requires relevance, any information unrelated to answering the user's "
    "question must fail the relevance criterion, even when that information is correct "
    "and present in the context.\n"
    "For relevance, do not ask whether information is generally related to the topic.\n"
    "Ask whether each piece of information is necessary to answer the specific user question.\n"
    "Information is irrelevant if the answer would remain complete and correct after removing it.\n"
    "Related, useful, grounded, or potentially helpful information can still be irrelevant.\n"
)

RETURN_INSTRUCTIONS: str = (
    "Return:\n"
    "- score: integer from 1 to 5\n"
    "- passed: true or false\n"
    "- reason: concise explanation\n"
    "- failed_criteria: names of failed rubric criteria\n"
    "If any rubric criterion fails:\n"
    "- passed must be false\n"
    "- failed_criteria must contain every failed criterion\n"
    "passed may be true only when failed_criteria is empty."
)

CRITERION_RETURN_INSTRUCTIONS: str = (
    "Return one result for every rubric criterion.\n"
    "For each criterion return:\n"
    "- name: exactly the criterion name from the 'Rubric criteria' list above\n"
    "- passed: true or false\n"
    "- reason: concise explanation\n"
    "Evaluate every criterion independently.\n"
    "Do not omit, rename, or add criteria.\n"
    "Do not use the rubric name as a criterion name."
)


def evaluate_with_judge(
    judge_input: JudgeInput,
    rubric: Rubric,
    judge: Judge,
) -> JudgeResult:
    return judge(judge_input, rubric)


def summarize_criterion_judge_result(
    result: CriterionJudgeResult,
) -> CriterionJudgeSummary:
    failed_criteria = []

    for criterion in result.criteria:
        if not criterion.passed:
            failed_criteria.append(criterion.name)

    passed = failed_criteria == []

    return CriterionJudgeSummary(
        passed=passed,
        failed_criteria=failed_criteria,
    )


def evaluate_with_criterion_judge(
    judge_input: JudgeInput,
    rubric: Rubric,
    judge: CriterionJudge,
) -> CriterionJudgeSummary:
    criterion_results = judge(judge_input, rubric)
    summary = summarize_criterion_judge_result(criterion_results)

    return summary


def build_judge_prompt(
    judge_input: JudgeInput,
    rubric: Rubric,
) -> str:
    lines: list[str] = []
    lines.append(JUDGE_INSTRUCTIONS)
    lines.append("")
    lines.append("Rubric:")
    lines.append(rubric["name"])

    for criterion in rubric["criteria"]:
        lines.append(f"- {criterion['name']}: {criterion['description']}")

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


def build_criterion_judge_prompt(
    judge_input: JudgeInput,
    rubric: Rubric,
) -> str:
    lines: list[str] = []
    lines.append(JUDGE_INSTRUCTIONS)
    lines.append("")
    lines.append("Rubric name:")
    lines.append(rubric["name"])
    lines.append("")
    lines.append("Rubric criteria:")

    for criterion in rubric["criteria"]:
        lines.append(f"- {criterion['name']}: {criterion['description']}")

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
    lines.append(CRITERION_RETURN_INSTRUCTIONS)

    return "\n".join(lines)


def validate_criterion_judge_result(
    result: CriterionJudgeResult,
    rubric: Rubric,
) -> None:
    expected_criteria: list[str] = []
    actual_criteria: list[str] = []

    for rubric_criterion in rubric["criteria"]:
        expected_criteria.append(rubric_criterion["name"])

    for result_criterion in result.criteria:
        actual_criteria.append(result_criterion.name)

    if sorted(expected_criteria) != sorted(actual_criteria):
        raise ValueError(
            f"Actual and expected criteria do not match: "
            f"expected={expected_criteria}, actual={actual_criteria}"
        )
