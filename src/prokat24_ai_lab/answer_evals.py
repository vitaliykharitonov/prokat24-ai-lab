import json
from pathlib import Path
from typing import TypedDict

from prokat24_ai_lab.answer_engine import answer_question
from prokat24_ai_lab.knowledge_base import KnowledgePage


class AnswerEvalCase(TypedDict):
    question: str
    expected_answer_type: str
    expected_source_url: str | None
    expected_answer_contains: list[str]


class AnswerEvalResult(TypedDict):
    question: str
    expected_answer_type: str
    actual_answer_type: str
    expected_source_url: str | None
    actual_source_url: str | None
    expected_answer_contains: list[str]
    missing_answer_fragments: list[str]
    passed: bool
    answer: str


def load_answer_eval_cases(input_file: Path) -> list[AnswerEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        return json.load(file)


def evaluate_answer_case(case: AnswerEvalCase, pages: list[KnowledgePage]) -> AnswerEvalResult:
    question = case["question"]
    answer = answer_question(question, pages)
    missing_answer_fragments = []

    actual_source_url = None if not answer["sources"] else answer["sources"][0]["url"]

    for fragment in case["expected_answer_contains"]:
        if fragment not in answer["answer"]:
            missing_answer_fragments.append(fragment)

    passed = (
        answer["answer_type"] == case["expected_answer_type"]
        and actual_source_url == case["expected_source_url"]
        and not missing_answer_fragments
    )

    return {
        "question": question,
        "expected_answer_type": case["expected_answer_type"],
        "actual_answer_type": answer["answer_type"],
        "expected_source_url": case["expected_source_url"],
        "actual_source_url": actual_source_url,
        "expected_answer_contains": case["expected_answer_contains"],
        "missing_answer_fragments": missing_answer_fragments,
        "passed": passed,
        "answer": answer["answer"],
    }
