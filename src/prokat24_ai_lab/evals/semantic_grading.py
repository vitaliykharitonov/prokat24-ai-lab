from collections.abc import Callable
from typing import TypedDict


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


class JudgeResult(TypedDict):
    score: int
    passed: bool
    reason: str
    failed_criteria: list[str]


Judge = Callable[[JudgeInput, Rubric], JudgeResult]
