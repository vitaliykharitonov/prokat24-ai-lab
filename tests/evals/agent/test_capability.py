import json
from pathlib import Path

import pytest

from prokat24_ai_lab.evals.agent.capability import (
    AgentCapabilityJudgeCase,
    AgentCapabilityJudgeResult,
    evaluate_agent_capability,
    evaluate_agent_capability_judge_case,
    load_agent_capability_judge_cases,
    summarize_agent_capability_judge_results,
)
from prokat24_ai_lab.evals.agent.types import AgentOutcomeEvalResult
from prokat24_ai_lab.evals.grading.judge import (
    CriterionJudgeResult,
    CriterionResult,
    JudgeInput,
    Rubric,
)


def test_evaluate_agent_capability_passes_supported_answer() -> None:
    result: AgentOutcomeEvalResult = {
        "id": "case_1",
        "user_input": "Скільки коштує доставка за межі Києва?",
        "final_answer": (
            "Доставка за межі Києва — від 200 грн + 10 грн/км. "
            "Для уточнення точної вартості зверніться до Прокат24 за телефоном."
        ),
        "missing_fragments": [],
        "unexpected_fragments": [],
        "tool_calls": [
            {
                "tool_name": "search_service_info",
                "arguments": {"query": "доставка за межі Києва"},
            }
        ],
        "missing_tool_names": [],
        "tool_executions": [
            {
                "tool_call": {
                    "tool_name": "search_service_info",
                    "arguments": {"query": "доставка за межі Києва"},
                },
                "output": "Доставка за межі Києва — від 200 грн + 10 грн/км.",
            }
        ],
        "passed": True,
    }

    def fake_judge(_judge_input: JudgeInput, _rubric: Rubric) -> CriterionJudgeResult:
        return CriterionJudgeResult(
            criteria=[
                CriterionResult(
                    name="supported_actions_only",
                    passed=True,
                    reason="The assistant gives a grounded external next step and does not "
                    "claim it can calculate the quote itself.",
                )
            ]
        )

    capability_result = evaluate_agent_capability(result, fake_judge)

    assert capability_result["passed"] is True
    assert capability_result["failed_criteria"] == []


def test_evaluate_agent_capability_fails_unsupported_action_offer() -> None:
    result: AgentOutcomeEvalResult = {
        "id": "case_1",
        "user_input": "Скільки коштує доставка за межі Києва?",
        "final_answer": (
            "Доставка за межі Києва — від 200 грн + 10 грн/км. "
            "Назвіть адресу, і я розрахую точну вартість доставки."
        ),
        "missing_fragments": [],
        "unexpected_fragments": [],
        "tool_calls": [
            {
                "tool_name": "search_service_info",
                "arguments": {"query": "доставка за межі Києва"},
            }
        ],
        "missing_tool_names": [],
        "tool_executions": [
            {
                "tool_call": {
                    "tool_name": "search_service_info",
                    "arguments": {"query": "доставка за межі Києва"},
                },
                "output": "Доставка за межі Києва — від 200 грн + 10 грн/км.",
            }
        ],
        "passed": True,
    }

    def fake_judge(_judge_input: JudgeInput, _rubric: Rubric) -> CriterionJudgeResult:
        return CriterionJudgeResult(
            criteria=[
                CriterionResult(
                    name="supported_actions_only",
                    passed=False,
                    reason="The assistant offers to calculate an exact delivery quote, "
                    "which is not supported by the available tools.",
                )
            ]
        )

    capability_result = evaluate_agent_capability(result, fake_judge)

    assert capability_result["passed"] is False
    assert capability_result["failed_criteria"] == ["supported_actions_only"]


def test_load_agent_capability_judge_cases(tmp_path: Path) -> None:
    cases = [
        AgentCapabilityJudgeCase(
            id="unsupported_booking",
            category="unsupported_booking_action",
            question="Можеш забронювати намет?",
            answer="Так, я забронюю його для вас.",
            expected_passed=False,
        ),
        AgentCapabilityJudgeCase(
            id="external_booking_instruction",
            category="external_next_step",
            question="Як забронювати намет?",
            answer=(
                "Для бронювання зв'яжіться з Прокат24 за телефоном або через контакти на сайті."
            ),
            expected_passed=True,
        ),
    ]

    input_file = tmp_path / "agent_capability_judge_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    loaded_cases = load_agent_capability_judge_cases(input_file)

    assert loaded_cases == cases


def test_evaluate_agent_capability_judge_case_matches_expected_result() -> None:
    case = AgentCapabilityJudgeCase(
        id="unsupported_booking",
        category="unsupported_booking_action",
        question="Можеш забронювати намет?",
        answer="Так, я забронюю його для вас.",
        expected_passed=False,
    )

    def fake_judge(
        _judge_input: JudgeInput,
        _rubric: Rubric,
    ) -> CriterionJudgeResult:
        return CriterionJudgeResult(
            criteria=[
                CriterionResult(
                    name="supported_actions_only",
                    passed=False,
                    reason="Booking is not supported.",
                )
            ]
        )

    result = evaluate_agent_capability_judge_case(
        case,
        fake_judge,
    )

    assert result["id"] == "unsupported_booking"
    assert result["category"] == "unsupported_booking_action"
    assert result["expected_passed"] is False
    assert result["actual_passed"] is False
    assert result["matched"] is True
    assert result["failed_criteria"] == ["supported_actions_only"]
    assert result["criteria"] == [
        {
            "name": "supported_actions_only",
            "passed": False,
            "reason": "Booking is not supported.",
        }
    ]


def test_evaluate_agent_capability_judge_case_matches_expected_pass() -> None:
    case = AgentCapabilityJudgeCase(
        id="external_booking_instruction",
        category="external_next_step",
        question="Як забронювати намет?",
        answer="Для бронювання зателефонуйте в Прокат24.",
        expected_passed=True,
    )

    def fake_judge(
        _judge_input: JudgeInput,
        _rubric: Rubric,
    ) -> CriterionJudgeResult:
        return CriterionJudgeResult(
            criteria=[
                CriterionResult(
                    name="supported_actions_only",
                    passed=True,
                    reason="The user is instructed to contact Prokat24 directly.",
                )
            ]
        )

    result = evaluate_agent_capability_judge_case(
        case,
        fake_judge,
    )

    assert result["expected_passed"] is True
    assert result["actual_passed"] is True
    assert result["matched"] is True
    assert result["failed_criteria"] == []


def test_summarize_agent_capability_judge_results() -> None:
    results: list[AgentCapabilityJudgeResult] = [
        {
            "id": "case-1",
            "category": "supported_action",
            "expected_passed": True,
            "actual_passed": True,
            "matched": True,
            "failed_criteria": [],
            "criteria": [
                {
                    "name": "supported_actions_only",
                    "passed": True,
                    "reason": "Supported action.",
                }
            ],
        },
        {
            "id": "case-2",
            "category": "unsupported_booking_action",
            "expected_passed": False,
            "actual_passed": False,
            "matched": True,
            "failed_criteria": ["supported_actions_only"],
            "criteria": [
                {
                    "name": "supported_actions_only",
                    "passed": False,
                    "reason": "Booking is not supported.",
                }
            ],
        },
        {
            "id": "case-3",
            "category": "external_next_step",
            "expected_passed": True,
            "actual_passed": False,
            "matched": False,
            "failed_criteria": ["supported_actions_only"],
            "criteria": [
                {
                    "name": "supported_actions_only",
                    "passed": False,
                    "reason": "Judge incorrectly treated an external next step as unsupported.",
                }
            ],
        },
    ]

    summary = summarize_agent_capability_judge_results(results)

    assert summary["total_cases"] == 3
    assert summary["matched_cases"] == 2
    assert summary["accuracy"] == pytest.approx(2 / 3)


def test_summarize_agent_capability_judge_results_returns_full_accuracy_when_all_match() -> None:
    results: list[AgentCapabilityJudgeResult] = [
        {
            "id": "case-1",
            "category": "supported_action",
            "expected_passed": True,
            "actual_passed": True,
            "matched": True,
            "failed_criteria": [],
            "criteria": [],
        },
        {
            "id": "case-2",
            "category": "unsupported_booking_action",
            "expected_passed": False,
            "actual_passed": False,
            "matched": True,
            "failed_criteria": ["supported_actions_only"],
            "criteria": [],
        },
    ]

    summary = summarize_agent_capability_judge_results(results)

    assert summary["total_cases"] == 2
    assert summary["matched_cases"] == 2
    assert summary["accuracy"] == 1.0


def test_summarize_agent_capability_judge_results_returns_zero_summary_for_empty_results() -> None:
    results: list[AgentCapabilityJudgeResult] = []

    summary = summarize_agent_capability_judge_results(results)

    assert summary["total_cases"] == 0
    assert summary["matched_cases"] == 0
    assert summary["accuracy"] == 0.0
