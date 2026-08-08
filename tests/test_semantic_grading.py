from prokat24_ai_lab.evals.semantic_grading import (
    JudgeInput,
    JudgeResult,
    Rubric,
    build_judge_prompt,
    evaluate_with_judge,
)


def test_evaluate_with_judge() -> None:
    judge_input: JudgeInput = {
        "question": "Чи безкоштовна доставка?",
        "context": "Доставка по Києву — від 200 грн.",
        "answer": "Доставка по Києву коштує від 200 грн.",
    }
    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "delivery_correctness",
                "description": "The answer must correctly state the delivery price.",
            }
        ],
    }

    judge_result: JudgeResult = {
        "score": 5,
        "passed": True,
        "reason": "The answer correctly states the delivery price.",
        "failed_criteria": [],
    }
    received_arguments: list[tuple[JudgeInput, Rubric]] = []

    def fake_judge(input_data: JudgeInput, received_rubric: Rubric) -> JudgeResult:
        received_arguments.append((input_data, received_rubric))
        return judge_result

    result = evaluate_with_judge(judge_input, rubric, fake_judge)

    assert result == judge_result
    assert received_arguments == [
        (
            {
                "question": "Чи безкоштовна доставка?",
                "context": "Доставка по Києву — від 200 грн.",
                "answer": "Доставка по Києву коштує від 200 грн.",
            },
            {
                "name": "delivery_answer",
                "criteria": [
                    {
                        "name": "delivery_correctness",
                        "description": "The answer must correctly state the delivery price.",
                    }
                ],
            },
        )
    ]


def test_build_judge_prompt_includes_evaluation_data() -> None:
    judge_input: JudgeInput = {
        "question": "Чи безкоштовна доставка?",
        "context": "Доставка по Києву — від 200 грн.",
        "answer": "Доставка по Києву коштує від 200 грн.",
    }
    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "delivery_correctness",
                "description": "The answer must correctly state the delivery price.",
            },
            {
                "name": "groundedness",
                "description": "The answer must not introduce facts unsupported by the context",
            },
        ],
    }

    prompt = build_judge_prompt(judge_input, rubric)

    assert "Чи безкоштовна доставка?" in prompt
    assert "Доставка по Києву — від 200 грн." in prompt
    assert "Доставка по Києву коштує від 200 грн." in prompt
    assert "delivery_answer" in prompt
    assert "delivery_correctness" in prompt
    assert "groundedness" in prompt
    assert "The answer must correctly state the delivery price." in prompt
    assert "The answer must not introduce facts unsupported by the context" in prompt
    assert "Do not use outside knowledge." in prompt
