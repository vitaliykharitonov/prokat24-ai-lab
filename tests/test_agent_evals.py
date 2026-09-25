import json
from pathlib import Path

import pytest

from prokat24_ai_lab.agents.types import AgentRunResult, ToolCall, ToolExecution
from prokat24_ai_lab.evals.agent_evals import (
    AgentCapabilityJudgeCase,
    AgentCapabilityJudgeResult,
    AgentCriterionCaseEvaluation,
    AgentCriterionEvalResult,
    AgentEvalCaseResult,
    AgentEvalConfiguration,
    AgentEvalReport,
    AgentEvalSummary,
    AgentGroundingEvalResult,
    AgentMetricSummary,
    AgentOutcomeCaseEvaluation,
    AgentOutcomeEvalCase,
    AgentOutcomeEvalResult,
    build_agent_eval_report,
    build_agent_grounding_context,
    compare_agent_eval_reports,
    compare_agent_metric_results,
    compare_latest_agent_eval_reports,
    evaluate_agent_capability,
    evaluate_agent_capability_judge_case,
    evaluate_agent_grounding,
    evaluate_agent_outcome_case,
    get_latest_agent_eval_report_files,
    list_agent_eval_report_files,
    load_agent_capability_judge_cases,
    load_agent_eval_report,
    load_agent_outcome_eval_cases,
    normalize_eval_text,
    run_agent_outcome_evals,
    save_agent_eval_report,
    summarize_agent_capability_judge_results,
)
from prokat24_ai_lab.evals.semantic_grading import (
    CriterionJudgeResult,
    CriterionResult,
    JudgeInput,
    Rubric,
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("① test", "1 test"),
        ("ЗАСТАВА", "застава"),
        ("2–місний намет", "2-місний намет"),
        ("2—місний намет", "2-місний намет"),
        ("4 000 грн", "4000 грн"),
        ("10 000 000 грн", "10000000 грн"),
        ("  Застава   4000 грн\nабо паспорт  ", "застава 4000 грн або паспорт"),
        ("4\u00a0000 грн", "4000 грн"),
    ],
)
def test_normalize_eval_text(text: str, expected: str) -> None:
    assert normalize_eval_text(text) == expected


def test_evaluate_agent_outcome_case_passes_when_expected_fragments_are_present() -> None:
    case = AgentOutcomeEvalCase(
        id="case_1",
        user_input="Скільки буде коштувати 2-місний намет на 3 дні?",
        expected_answer_fragments=[
            "660 грн",
            "2-місн",
        ],
        forbidden_answer_fragments=[],
        required_tool_names=[],
    )

    agent_result = AgentRunResult(
        final_answer=("Вартість оренди 2-місного намету на 3 дні — 660 грн. Застава — 4000 грн."),
        tool_calls=[
            ToolCall(
                tool_name="search_products",
                arguments={"query": "2-місний намет"},
            )
        ],
        tool_executions=[
            {
                "tool_call": {
                    "tool_name": "search_products",
                    "arguments": {"query": "2-місний намет"},
                },
                "output": "Намет 2-місний",
            }
        ],
    )

    result = evaluate_agent_outcome_case(case, agent_result)

    assert result["missing_fragments"] == []
    assert result["passed"] is True
    assert result["tool_calls"] == agent_result["tool_calls"]
    assert result["tool_executions"] == agent_result["tool_executions"]
    assert result["user_input"] == case["user_input"]
    assert result["final_answer"] == agent_result["final_answer"]


def test_evaluate_agent_outcome_case_fails_when_expected_fragment_is_missing() -> None:
    case = AgentOutcomeEvalCase(
        id="case_1",
        user_input="Скільки буде коштувати 2-місний намет на 3 дні?",
        expected_answer_fragments=[
            "660 грн",
            "2-місн",
        ],
        forbidden_answer_fragments=[],
        required_tool_names=[],
    )

    agent_result = AgentRunResult(
        final_answer="Вартість оренди 2-місного намету на 3 дні — 700 грн.",
        tool_calls=[
            ToolCall(
                tool_name="search_products",
                arguments={"query": "2-місний намет"},
            )
        ],
        tool_executions=[
            {
                "tool_call": {
                    "tool_name": "search_products",
                    "arguments": {"query": "2-місний намет"},
                },
                "output": "Намет 2-місний",
            }
        ],
    )

    result = evaluate_agent_outcome_case(case, agent_result)

    assert result["missing_fragments"] == ["660 грн"]
    assert result["passed"] is False
    assert result["tool_calls"] == agent_result["tool_calls"]
    assert result["tool_executions"] == agent_result["tool_executions"]
    assert result["user_input"] == case["user_input"]
    assert result["final_answer"] == agent_result["final_answer"]


def test_evaluate_agent_outcome_case_fails_when_forbidden_fragment_is_present() -> None:
    case = AgentOutcomeEvalCase(
        id="case_1",
        user_input="Яка застава за 2-місний намет?",
        expected_answer_fragments=["4000 грн"],
        forbidden_answer_fragments=[
            "без застави",
            "застава не потрібна",
        ],
        required_tool_names=[],
    )

    agent_result = AgentRunResult(
        final_answer=("Застава за 2-місний намет — 4000 грн. Але застава не потрібна."),
        tool_calls=[
            ToolCall(
                tool_name="search_products",
                arguments={"query": "2-місний намет"},
            )
        ],
        tool_executions=[
            {
                "tool_call": {
                    "tool_name": "search_products",
                    "arguments": {"query": "2-місний намет"},
                },
                "output": "Намет 2-місний",
            }
        ],
    )

    result = evaluate_agent_outcome_case(case, agent_result)

    assert result["missing_fragments"] == []
    assert result["unexpected_fragments"] == ["застава не потрібна"]
    assert result["passed"] is False


def test_evaluate_agent_outcome_case_passes_when_required_tool_is_called() -> None:
    case = AgentOutcomeEvalCase(
        id="case_1",
        user_input="Знайди 2-місний намет",
        expected_answer_fragments=["2-місн"],
        forbidden_answer_fragments=[],
        required_tool_names=["search_products"],
    )

    agent_result = AgentRunResult(
        final_answer="Знайдено 2-місний намет.",
        tool_calls=[
            ToolCall(
                tool_name="search_products",
                arguments={"query": "2-місний намет"},
            )
        ],
        tool_executions=[
            {
                "tool_call": {
                    "tool_name": "search_products",
                    "arguments": {"query": "2-місний намет"},
                },
                "output": "Намет 2-місний",
            }
        ],
    )

    result = evaluate_agent_outcome_case(case, agent_result)

    assert result["missing_fragments"] == []
    assert result["unexpected_fragments"] == []
    assert result["missing_tool_names"] == []
    assert result["passed"] is True


def test_evaluate_agent_outcome_case_fails_when_required_tool_is_missing() -> None:
    case = AgentOutcomeEvalCase(
        id="case_1",
        user_input="Знайди 2-місний намет",
        expected_answer_fragments=["2-місн"],
        forbidden_answer_fragments=[],
        required_tool_names=["search_products"],
    )

    agent_result = AgentRunResult(
        final_answer="Можу допомогти знайти 2-місний намет.",
        tool_calls=[],
        tool_executions=[],
    )

    result = evaluate_agent_outcome_case(case, agent_result)

    assert result["missing_fragments"] == []
    assert result["unexpected_fragments"] == []
    assert result["missing_tool_names"] == ["search_products"]
    assert result["passed"] is False


def test_evaluate_agent_outcome_case_normalizes_answer_fragments() -> None:
    case = AgentOutcomeEvalCase(
        id="case_1",
        user_input="Яка застава за 2-місний намет?",
        expected_answer_fragments=["4000 грн"],
        forbidden_answer_fragments=[],
        required_tool_names=["search_products"],
    )

    agent_result = AgentRunResult(
        final_answer="Застава за намет — 4 000 грн.",
        tool_calls=[
            ToolCall(
                tool_name="search_products",
                arguments={"query": "2-місний намет"},
            )
        ],
        tool_executions=[
            {
                "tool_call": {
                    "tool_name": "search_products",
                    "arguments": {"query": "2-місний намет"},
                },
                "output": "Намет 2-місний",
            }
        ],
    )

    result = evaluate_agent_outcome_case(case, agent_result)

    assert result["missing_fragments"] == []
    assert result["passed"] is True


def test_run_agent_outcome_evals_returns_results_for_all_cases() -> None:
    cases = [
        AgentOutcomeEvalCase(
            id="case_1",
            user_input="Скільки буде коштувати 2-місний намет на 3 дні?",
            expected_answer_fragments=["660 грн", "2-місн"],
            forbidden_answer_fragments=[],
            required_tool_names=[],
        ),
        AgentOutcomeEvalCase(
            id="case_2",
            user_input="Знайди спальник",
            expected_answer_fragments=["спальник"],
            forbidden_answer_fragments=[],
            required_tool_names=[],
        ),
    ]

    def fake_agent_runner(user_input: str) -> AgentRunResult:
        if "2-місний намет" in user_input:
            return AgentRunResult(
                final_answer="Вартість оренди 2-місного намету на 3 дні — 660 грн.",
                tool_calls=[
                    ToolCall(
                        tool_name="search_products",
                        arguments={"query": "2-місний намет"},
                    )
                ],
                tool_executions=[
                    {
                        "tool_call": {
                            "tool_name": "search_products",
                            "arguments": {"query": "2-місний намет"},
                        },
                        "output": "Намет 2-місний",
                    }
                ],
            )

        return AgentRunResult(
            final_answer="Знайдено туристичний спальник.",
            tool_calls=[
                ToolCall(
                    tool_name="search_products",
                    arguments={"query": "спальник"},
                )
            ],
            tool_executions=[
                {
                    "tool_call": {
                        "tool_name": "search_products",
                        "arguments": {"query": "спальник"},
                    },
                    "output": "Туристичний спальник",
                }
            ],
        )

    results = run_agent_outcome_evals(cases, fake_agent_runner)

    assert len(results) == 2
    assert results[0]["passed"] is True
    assert results[1]["passed"] is True
    assert results[0]["tool_executions"]
    assert results[1]["tool_executions"]


def test_load_agent_outcome_eval_cases(tmp_path: Path) -> None:
    cases = [
        AgentOutcomeEvalCase(
            id="case_1",
            user_input="Скільки буде коштувати 2-місний намет на 3 дні?",
            expected_answer_fragments=[
                "660 грн",
                "2-місн",
            ],
            forbidden_answer_fragments=[],
            required_tool_names=[],
        ),
        AgentOutcomeEvalCase(
            id="case_2",
            user_input="Знайди 2-місний намет",
            expected_answer_fragments=[
                "2-місн",
            ],
            forbidden_answer_fragments=[],
            required_tool_names=[],
        ),
    ]

    input_file = tmp_path / "agent_outcome_eval_cases.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(cases, file, ensure_ascii=False, indent=2)

    loaded_cases = load_agent_outcome_eval_cases(input_file)

    assert loaded_cases == cases


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


def test_build_agent_grounding_context() -> None:
    tool_executions: list[ToolExecution] = [
        {
            "tool_call": {
                "tool_name": "search_service_info",
                "arguments": {"query": "доставка по Києву"},
            },
            "output": "Доставка по Києву — від 200 грн.",
        }
    ]

    context = build_agent_grounding_context(tool_executions)

    assert context == (
        "[Tool execution 1]\n"
        "Tool: search_service_info\n"
        "Arguments: {'query': 'доставка по Києву'}\n"
        "Output:\n"
        "Доставка по Києву — від 200 грн.\n"
    )


def test_evaluate_agent_grounding_passes_supported_answer() -> None:
    result: AgentOutcomeEvalResult = {
        "id": "case_1",
        "user_input": "Скільки коштує доставка по Києву?",
        "final_answer": "Доставка по Києву — від 200 грн.",
        "missing_fragments": [],
        "unexpected_fragments": [],
        "tool_calls": [
            {
                "tool_name": "search_service_info",
                "arguments": {"query": "доставка по Києву"},
            }
        ],
        "tool_executions": [
            {
                "tool_call": {
                    "tool_name": "search_service_info",
                    "arguments": {"query": "доставка по Києву"},
                },
                "output": "Доставка по Києву — від 200 грн.",
            }
        ],
        "missing_tool_names": [],
        "passed": True,
    }

    def fake_judge(
        _judge_input: JudgeInput,
        _rubric: Rubric,
    ) -> CriterionJudgeResult:
        return CriterionJudgeResult(
            criteria=[
                CriterionResult(
                    name="grounded_facts_only",
                    passed=True,
                    reason="The delivery price is directly supported by the tool output.",
                )
            ]
        )

    grounding_result = evaluate_agent_grounding(result, fake_judge)

    assert grounding_result["passed"] is True
    assert grounding_result["failed_criteria"] == []


def test_evaluate_agent_grounding_fails_unsupported_fact() -> None:
    result: AgentOutcomeEvalResult = {
        "id": "case_1",
        "user_input": "Скільки коштує доставка по Києву?",
        "final_answer": "Доставка по Києву — 300 грн.",
        "missing_fragments": [],
        "unexpected_fragments": [],
        "tool_calls": [
            {
                "tool_name": "search_service_info",
                "arguments": {"query": "доставка по Києву"},
            }
        ],
        "tool_executions": [
            {
                "tool_call": {
                    "tool_name": "search_service_info",
                    "arguments": {"query": "доставка по Києву"},
                },
                "output": "Доставка по Києву — від 200 грн.",
            }
        ],
        "missing_tool_names": [],
        "passed": False,
    }

    def fake_judge(
        _judge_input: JudgeInput,
        _rubric: Rubric,
    ) -> CriterionJudgeResult:
        return CriterionJudgeResult(
            criteria=[
                CriterionResult(
                    name="grounded_facts_only",
                    passed=False,
                    reason=(
                        "The answer states 300 UAH, but the tool output only supports "
                        "a price starting from 200 UAH."
                    ),
                )
            ]
        )

    grounding_result = evaluate_agent_grounding(result, fake_judge)

    assert grounding_result["passed"] is False
    assert grounding_result["failed_criteria"] == ["grounded_facts_only"]


def test_build_agent_eval_report() -> None:
    outcome_results: list[AgentOutcomeEvalResult] = [
        AgentOutcomeEvalResult(
            id="kyiv_delivery",
            user_input="Скільки коштує доставка по Києву?",
            final_answer="Доставка по Києву — від 200 грн.",
            missing_fragments=[],
            unexpected_fragments=[],
            tool_calls=[
                {
                    "tool_name": "search_service_info",
                    "arguments": {"query": "доставка по Києву"},
                }
            ],
            missing_tool_names=[],
            tool_executions=[
                {
                    "tool_call": {
                        "tool_name": "search_service_info",
                        "arguments": {"query": "доставка по Києву"},
                    },
                    "output": "Доставка по Києву — від 200 грн.",
                }
            ],
            passed=True,
        )
    ]

    capability_results: list[AgentCriterionEvalResult] = [
        AgentCriterionEvalResult(
            user_input="Скільки коштує доставка по Києву?",
            final_answer="Доставка по Києву — від 200 грн.",
            passed=True,
            failed_criteria=[],
            criteria=[
                {
                    "name": "supported_actions_only",
                    "passed": True,
                    "reason": "The answer stays within supported capabilities.",
                }
            ],
        )
    ]

    grounding_results: list[AgentGroundingEvalResult] = [
        AgentGroundingEvalResult(
            user_input="Скільки коштує доставка по Києву?",
            final_answer="Доставка по Києву — від 200 грн.",
            passed=True,
            failed_criteria=[],
            criteria=[
                {
                    "name": "grounded_facts_only",
                    "passed": True,
                    "reason": "The delivery price is supported by the tool output.",
                }
            ],
        )
    ]

    configuration = AgentEvalConfiguration(
        agent_model="gpt-5-mini",
        judge_model="gpt-5.4-mini-2026-03-17",
        max_agent_steps=5,
        eval_cases_file="data/agent_outcome_eval_cases.json",
    )

    report = build_agent_eval_report(
        outcome_results=outcome_results,
        capability_results=capability_results,
        grounding_results=grounding_results,
        configuration=configuration,
        run_id="20260923T120000Z_abc1234",
        created_at="2026-09-23T12:00:00+00:00",
        git_commit="abc1234",
        git_dirty=False,
    )

    assert report["report_type"] == "agent_eval_report"
    assert report["schema_version"] == 1
    assert report["created_at"] == "2026-09-23T12:00:00+00:00"
    assert report["git_commit"] == "abc1234"
    assert report["git_dirty"] is False
    assert report["run_id"] == "20260923T120000Z_abc1234"

    assert report["configuration"] == configuration

    assert report["summary"] == {
        "outcome": {
            "total_cases": 1,
            "passed_cases": 1,
            "pass_rate": 1.0,
        },
        "capability": {
            "total_cases": 1,
            "passed_cases": 1,
            "pass_rate": 1.0,
        },
        "grounding": {
            "total_cases": 1,
            "passed_cases": 1,
            "pass_rate": 1.0,
        },
    }

    assert len(report["results"]) == 1

    case_result = report["results"][0]

    assert case_result["id"] == "kyiv_delivery"
    assert case_result["outcome"]["passed"] is True
    assert case_result["capability"]["passed"] is True
    assert case_result["grounding"]["passed"] is True


def test_save_agent_eval_report_saves_historical_and_latest_files(tmp_path: Path) -> None:
    report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-23T18:00:00+00:00",
        git_commit="abc1234",
        git_dirty=False,
        run_id="20260923T180000Z_abc1234",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
            capability=AgentMetricSummary(
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
            grounding=AgentMetricSummary(
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
        ),
        results=[],
    )
    output_dir = tmp_path / "agent"
    historical_file = output_dir / "runs" / "20260923T180000Z_abc1234.json"
    latest_file = output_dir / "latest.json"

    saved_historical_file, saved_latest_file = save_agent_eval_report(report, output_dir)

    with open(historical_file, encoding="utf-8") as file:
        historical_report = json.load(file)

    with open(latest_file, encoding="utf-8") as file:
        latest_report = json.load(file)

    assert historical_report == latest_report == report
    assert saved_historical_file == historical_file
    assert saved_latest_file == latest_file


def test_compare_agent_metric_results_classifies_case_changes() -> None:
    previous_report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-23T18:00:00+00:00",
        git_commit="abc1234",
        git_dirty=False,
        run_id="previous_run",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(
                total_cases=4,
                passed_cases=2,
                pass_rate=0.5,
            ),
            capability=AgentMetricSummary(
                total_cases=0,
                passed_cases=0,
                pass_rate=0.0,
            ),
            grounding=AgentMetricSummary(
                total_cases=0,
                passed_cases=0,
                pass_rate=0.0,
            ),
        ),
        results=[
            {
                "id": "improved_case",
                "outcome": {"passed": False},
            },
            {
                "id": "worsened_case",
                "outcome": {"passed": True},
            },
            {
                "id": "unchanged_case",
                "outcome": {"passed": True},
            },
            {
                "id": "removed_case",
                "outcome": {"passed": False},
            },
        ],
    )
    current_report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-24T18:00:00+00:00",
        git_commit="def5678",
        git_dirty=False,
        run_id="current_run",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(
                total_cases=4,
                passed_cases=3,
                pass_rate=0.75,
            ),
            capability=AgentMetricSummary(
                total_cases=0,
                passed_cases=0,
                pass_rate=0.0,
            ),
            grounding=AgentMetricSummary(
                total_cases=0,
                passed_cases=0,
                pass_rate=0.0,
            ),
        ),
        results=[
            {
                "id": "improved_case",
                "outcome": {"passed": True},
            },
            {
                "id": "worsened_case",
                "outcome": {"passed": False},
            },
            {
                "id": "unchanged_case",
                "outcome": {"passed": True},
            },
            {
                "id": "added_case",
                "outcome": {"passed": True},
            },
        ],
    )

    comparison = compare_agent_metric_results(previous_report, current_report, "outcome")

    assert comparison["previous_pass_rate"] == 0.5
    assert comparison["current_pass_rate"] == 0.75
    assert comparison["pass_rate_delta"] == 0.25

    assert comparison["improved_case_ids"] == ["improved_case"]
    assert comparison["worsened_case_ids"] == ["worsened_case"]
    assert comparison["unchanged_case_ids"] == ["unchanged_case"]
    assert comparison["added_case_ids"] == ["added_case"]
    assert comparison["removed_case_ids"] == ["removed_case"]


def test_compare_agent_eval_reports_compares_all_metrics() -> None:
    previous_report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-23T18:00:00+00:00",
        git_commit="abc1234",
        git_dirty=False,
        run_id="previous_run",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(
                total_cases=1,
                passed_cases=0,
                pass_rate=0.0,
            ),
            capability=AgentMetricSummary(
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
            grounding=AgentMetricSummary(
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
        ),
        results=[
            AgentEvalCaseResult(
                id="test_case",
                user_input="Test question",
                final_answer="Test answer",
                tool_calls=[],
                tool_executions=[],
                outcome=AgentOutcomeCaseEvaluation(
                    passed=False,
                    missing_fragments=[],
                    unexpected_fragments=[],
                    missing_tool_names=[],
                ),
                capability=AgentCriterionCaseEvaluation(
                    passed=True,
                    failed_criteria=[],
                    criteria=[],
                ),
                grounding=AgentCriterionCaseEvaluation(
                    passed=True,
                    failed_criteria=[],
                    criteria=[],
                ),
            ),
        ],
    )
    current_report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-24T18:00:00+00:00",
        git_commit="def5678",
        git_dirty=False,
        run_id="current_run",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
            capability=AgentMetricSummary(
                total_cases=1,
                passed_cases=0,
                pass_rate=0.0,
            ),
            grounding=AgentMetricSummary(
                total_cases=1,
                passed_cases=1,
                pass_rate=1.0,
            ),
        ),
        results=[
            AgentEvalCaseResult(
                id="test_case",
                user_input="Test question",
                final_answer="Test answer",
                tool_calls=[],
                tool_executions=[],
                outcome=AgentOutcomeCaseEvaluation(
                    passed=True,
                    missing_fragments=[],
                    unexpected_fragments=[],
                    missing_tool_names=[],
                ),
                capability=AgentCriterionCaseEvaluation(
                    passed=False,
                    failed_criteria=["supported_actions_only"],
                    criteria=[],
                ),
                grounding=AgentCriterionCaseEvaluation(
                    passed=True,
                    failed_criteria=[],
                    criteria=[],
                ),
            ),
        ],
    )

    reports_comparison = compare_agent_eval_reports(previous_report, current_report)

    assert reports_comparison["previous_run_id"] == "previous_run"
    assert reports_comparison["current_run_id"] == "current_run"

    assert reports_comparison["outcome"]["improved_case_ids"] == ["test_case"]
    assert reports_comparison["outcome"]["pass_rate_delta"] == 1.0

    assert reports_comparison["capability"]["worsened_case_ids"] == ["test_case"]
    assert reports_comparison["capability"]["pass_rate_delta"] == -1.0

    assert reports_comparison["grounding"]["unchanged_case_ids"] == ["test_case"]
    assert reports_comparison["grounding"]["pass_rate_delta"] == 0.0


def test_load_agent_eval_report_reads_json_file(tmp_path: Path) -> None:
    report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-24T18:00:00+00:00",
        git_commit="def5678",
        git_dirty=False,
        run_id="current_run",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(
                total_cases=4,
                passed_cases=3,
                pass_rate=0.75,
            ),
            capability=AgentMetricSummary(
                total_cases=0,
                passed_cases=0,
                pass_rate=0.0,
            ),
            grounding=AgentMetricSummary(
                total_cases=0,
                passed_cases=0,
                pass_rate=0.0,
            ),
        ),
        results=[],
    )

    input_file = tmp_path / "agent_eval_report.json"

    with open(input_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    loaded_data = load_agent_eval_report(input_file)

    assert loaded_data == report


def test_list_agent_eval_report_files_returns_sorted_json_files(tmp_path: Path) -> None:
    reports_dir = tmp_path / "agent"
    runs_dir = reports_dir / "runs"
    runs_dir.mkdir(parents=True)

    first_file = runs_dir / "20260923T120000Z_aaa1111.json"
    second_file = runs_dir / "20260924T120000Z_bbb2222.json"
    third_file = runs_dir / "20260925T120000Z_ccc3333.json"
    ignored_file = runs_dir / "notes.txt"

    third_file.write_text("{}", encoding="utf-8")
    first_file.write_text("{}", encoding="utf-8")
    ignored_file.write_text("ignore", encoding="utf-8")
    second_file.write_text("{}", encoding="utf-8")

    report_files = list_agent_eval_report_files(reports_dir)

    assert report_files == [
        first_file,
        second_file,
        third_file,
    ]


def test_get_latest_agent_eval_report_files_returns_previous_and_current(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "agent"
    runs_dir = reports_dir / "runs"
    runs_dir.mkdir(parents=True)

    first_file = runs_dir / "20260923T120000Z_aaa1111.json"
    second_file = runs_dir / "20260924T120000Z_bbb2222.json"
    third_file = runs_dir / "20260925T120000Z_ccc3333.json"

    first_file.write_text("{}", encoding="utf-8")
    third_file.write_text("{}", encoding="utf-8")
    second_file.write_text("{}", encoding="utf-8")

    previous_file, current_file = get_latest_agent_eval_report_files(reports_dir)

    assert previous_file == second_file
    assert current_file == third_file


def test_get_latest_agent_eval_report_files_raises_for_less_than_two_reports(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "agent"
    runs_dir = reports_dir / "runs"
    runs_dir.mkdir(parents=True)

    only_file = runs_dir / "20260925T120000Z_ccc3333.json"
    only_file.write_text("{}", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="At least two agent eval reports are required for comparison",
    ):
        get_latest_agent_eval_report_files(reports_dir)


def test_compare_latest_agent_eval_reports_compares_two_latest_runs(
    tmp_path: Path,
) -> None:
    reports_dir = tmp_path / "agent"

    first_report = AgentEvalReport(
        report_type="agent_eval_report",
        schema_version=1,
        created_at="2026-09-23T12:00:00+00:00",
        git_commit="aaa1111",
        git_dirty=False,
        run_id="20260923T120000Z_aaa1111",
        configuration=AgentEvalConfiguration(
            agent_model="gpt-5-mini",
            judge_model="gpt-5.4-mini-2026-03-17",
            max_agent_steps=5,
            eval_cases_file="data/agent_outcome_eval_cases.json",
        ),
        summary=AgentEvalSummary(
            outcome=AgentMetricSummary(total_cases=0, passed_cases=0, pass_rate=0.0),
            capability=AgentMetricSummary(total_cases=0, passed_cases=0, pass_rate=0.0),
            grounding=AgentMetricSummary(total_cases=0, passed_cases=0, pass_rate=0.0),
        ),
        results=[],
    )

    second_report = first_report.copy()
    second_report["created_at"] = "2026-09-24T12:00:00+00:00"
    second_report["git_commit"] = "bbb2222"
    second_report["run_id"] = "20260924T120000Z_bbb2222"

    third_report = first_report.copy()
    third_report["created_at"] = "2026-09-25T12:00:00+00:00"
    third_report["git_commit"] = "ccc3333"
    third_report["run_id"] = "20260925T120000Z_ccc3333"

    save_agent_eval_report(first_report, reports_dir)
    save_agent_eval_report(third_report, reports_dir)
    save_agent_eval_report(second_report, reports_dir)

    comparison = compare_latest_agent_eval_reports(reports_dir)

    assert comparison["previous_run_id"] == "20260924T120000Z_bbb2222"
    assert comparison["current_run_id"] == "20260925T120000Z_ccc3333"
