from collections.abc import Callable
from typing import TypedDict


class RagAnswerEvalCase(TypedDict):
    question: str
    expected_answer_contains: list[str]


class RagAnswerEvalResult(TypedDict):
    question: str
    expected_answer_contains: list[str]
    missing_answer_fragments: list[str]
    passed: bool
    answer: str


RagAnswerer = Callable[[str], str]


def evaluate_rag_answer_case(
    case: RagAnswerEvalCase,
    answerer: RagAnswerer,
) -> RagAnswerEvalResult:
    question = case["question"]

    answer = answerer(question)

    expected_answer_contains = case["expected_answer_contains"]
    missing_answer_fragments = []

    for fragment in expected_answer_contains:
        if fragment.lower() not in answer.lower():
            missing_answer_fragments.append(fragment)

    passed = not missing_answer_fragments

    return {
        "question": case["question"],
        "expected_answer_contains": expected_answer_contains,
        "missing_answer_fragments": missing_answer_fragments,
        "passed": passed,
        "answer": answer,
    }


def run_rag_answer_evals(
    cases: list[RagAnswerEvalCase],
    answerer: RagAnswerer,
) -> list[RagAnswerEvalResult]:
    results: list[RagAnswerEvalResult] = []
    for case in cases:
        results.append(evaluate_rag_answer_case(case, answerer))

    return results
