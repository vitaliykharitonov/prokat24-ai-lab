import pytest

from prokat24_ai_lab.evals.answers.ragas import (
    RagasEvalCase,
    RagasEvalResult,
    build_ragas_eval_result,
    summarize_ragas_eval_results,
)


def test_build_ragas_eval_result() -> None:
    case: RagasEvalCase = {
        "question": "Які умови доставки за межі Києва?",
        "reference": "Доставка можлива за попередньою домовленістю.",
    }

    retrieved_contexts = [
        "Доставка можлива за попередньою домовленістю.",
        "Пункт прокату працює щодня.",
    ]

    result = build_ragas_eval_result(
        case=case,
        response="Доставка можлива за попередньою домовленістю.",
        retrieved_contexts=retrieved_contexts,
        faithfulness=1.0,
        answer_relevancy=0.8,
        context_precision=0.9,
        context_recall=0.7,
    )

    assert result == {
        "question": "Які умови доставки за межі Києва?",
        "reference": "Доставка можлива за попередньою домовленістю.",
        "response": "Доставка можлива за попередньою домовленістю.",
        "retrieved_contexts": retrieved_contexts,
        "faithfulness": 1.0,
        "answer_relevancy": 0.8,
        "context_precision": 0.9,
        "context_recall": 0.7,
    }


def test_summarize_ragas_eval_results_calculates_average_metrics() -> None:
    results: list[RagasEvalResult] = [
        {
            "question": "Question 1",
            "reference": "Reference 1",
            "response": "Response 1",
            "retrieved_contexts": ["Context 1"],
            "faithfulness": 1.0,
            "answer_relevancy": 0.8,
            "context_precision": 0.9,
            "context_recall": 1.0,
        },
        {
            "question": "Question 2",
            "reference": "Reference 2",
            "response": "Response 2",
            "retrieved_contexts": ["Context 2"],
            "faithfulness": 0.8,
            "answer_relevancy": 0.6,
            "context_precision": 0.7,
            "context_recall": 0.5,
        },
    ]
    expected = {
        "total_cases": 2,
        "average_faithfulness": pytest.approx(0.9),
        "average_answer_relevancy": pytest.approx(0.7),
        "average_context_precision": pytest.approx(0.8),
        "average_context_recall": pytest.approx(0.75),
    }

    summary = summarize_ragas_eval_results(results)

    assert summary == expected


def test_summarize_ragas_eval_results_ignores_missing_metrics() -> None:
    results: list[RagasEvalResult] = [
        {
            "question": "Question 1",
            "reference": "Reference 1",
            "response": "Response 1",
            "retrieved_contexts": ["Context 1"],
            "faithfulness": None,
            "answer_relevancy": 0.4,
            "context_precision": None,
            "context_recall": None,
        },
        {
            "question": "Question 2",
            "reference": "Reference 2",
            "response": "Response 2",
            "retrieved_contexts": ["Context 2"],
            "faithfulness": None,
            "answer_relevancy": 0.8,
            "context_precision": None,
            "context_recall": None,
        },
    ]
    expected = {
        "total_cases": 2,
        "average_faithfulness": None,
        "average_answer_relevancy": pytest.approx(0.6),
        "average_context_precision": None,
        "average_context_recall": None,
    }

    summary = summarize_ragas_eval_results(results)

    assert summary == expected
