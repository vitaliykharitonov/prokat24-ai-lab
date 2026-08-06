from prokat24_ai_lab.evals.rag_answer_evals import RagAnswerEvalCase, evaluate_rag_answer_case


def test_evaluate_rag_answer_case_passes_when_answer_contains_expected_fragments() -> None:
    case: RagAnswerEvalCase = {
        "question": "Як отримати спорядження, не приїжджаючи до вас?",
        "expected_answer_contains": [
            "доставк",
            "попередньою домовленістю",
        ],
    }
    received_questions: list[str] = []
    expected_answer = (
        "Доставка можлива за попередньою домовленістю. По Києву — від 200 грн [Document 1]."
    )

    def fake_answerer(question: str) -> str:
        received_questions.append(question)
        return expected_answer

    result = evaluate_rag_answer_case(case, fake_answerer)

    assert received_questions == [case["question"]]
    assert result["passed"] is True
    assert result["missing_answer_fragments"] == []
    assert result["answer"] == expected_answer


def test_evaluate_rag_answer_case_reports_missing_fragments() -> None:
    case: RagAnswerEvalCase = {
        "question": "Як отримати спорядження, не приїжджаючи до вас?",
        "expected_answer_contains": [
            "доставк",
            "безкоштовно",
        ],
    }
    received_questions: list[str] = []
    expected_answer = (
        "Доставка можлива за попередньою домовленістю. По Києву — від 200 грн [Document 1]."
    )

    def fake_answerer(question: str) -> str:
        received_questions.append(question)
        return expected_answer

    result = evaluate_rag_answer_case(case, fake_answerer)

    assert received_questions == [case["question"]]
    assert result["answer"] == expected_answer
    assert result["passed"] is False
    assert result["missing_answer_fragments"] == ["безкоштовно"]
