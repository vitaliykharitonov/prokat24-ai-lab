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

    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None
    context_recall: float | None


class RagasEvalSummary(TypedDict):
    total_cases: int
    average_faithfulness: float | None
    average_answer_relevancy: float | None
    average_context_precision: float | None
    average_context_recall: float | None


def load_ragas_eval_cases(input_file: Path) -> list[RagasEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        loaded_data: list[RagasEvalCase] = json.load(file)

    return loaded_data


def build_ragas_eval_result(
    case: RagasEvalCase,
    response: str,
    retrieved_contexts: list[str],
    faithfulness: float | None,
    answer_relevancy: float | None,
    context_precision: float | None,
    context_recall: float | None,
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


def summarize_ragas_eval_results(
    results: list[RagasEvalResult],
) -> RagasEvalSummary:
    total_cases = len(results)

    average_faithfulness: float | None = None
    average_answer_relevancy: float | None = None
    average_context_precision: float | None = None
    average_context_recall: float | None = None

    faithfulness_values: list[float] = []
    answer_relevancy_values: list[float] = []
    context_precision_values: list[float] = []
    context_recall_values: list[float] = []

    for result in results:
        if result["faithfulness"] is not None:
            faithfulness_values.append(result["faithfulness"])

        if result["answer_relevancy"] is not None:
            answer_relevancy_values.append(result["answer_relevancy"])

        if result["context_precision"] is not None:
            context_precision_values.append(result["context_precision"])

        if result["context_recall"] is not None:
            context_recall_values.append(result["context_recall"])

    if faithfulness_values:
        average_faithfulness = sum(faithfulness_values) / len(faithfulness_values)

    if answer_relevancy_values:
        average_answer_relevancy = sum(answer_relevancy_values) / len(answer_relevancy_values)

    if context_precision_values:
        average_context_precision = sum(context_precision_values) / len(context_precision_values)

    if context_recall_values:
        average_context_recall = sum(context_recall_values) / len(context_recall_values)

    return RagasEvalSummary(
        total_cases=total_cases,
        average_faithfulness=average_faithfulness,
        average_answer_relevancy=average_answer_relevancy,
        average_context_precision=average_context_precision,
        average_context_recall=average_context_recall,
    )
