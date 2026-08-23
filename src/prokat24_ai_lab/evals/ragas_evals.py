import json
from pathlib import Path
from typing import TypedDict


class RagasEvalCase(TypedDict):
    question: str
    reference: str


class RagasEvalResult(TypedDict):
    question: str
    reference: str
    response: str
    retrieved_contexts: list[str]

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_ragas_eval_cases(input_file: Path) -> list[RagasEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        loaded_data: list[RagasEvalCase] = json.load(file)

    return loaded_data


def build_ragas_eval_result(
    case: RagasEvalCase,
    response: str,
    retrieved_contexts: list[str],
    faithfulness: float,
    answer_relevancy: float,
    context_precision: float,
    context_recall: float,
) -> RagasEvalResult:
    return RagasEvalResult(
        question=case["question"],
        reference=case["reference"],
        response=response,
        retrieved_contexts=retrieved_contexts,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
        context_recall=context_recall,
    )
