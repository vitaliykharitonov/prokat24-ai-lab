from prokat24_ai_lab.evals.ragas_evals import (
    RagasEvalCase,
    build_ragas_eval_result,
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
