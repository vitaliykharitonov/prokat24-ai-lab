import json
from pathlib import Path

import pytest

from prokat24_ai_lab.agents.types import AgentRunResult, ToolCall
from prokat24_ai_lab.evals.agent_evals import (
    AgentOutcomeEvalCase,
    AgentOutcomeEvalResult,
    evaluate_agent_outcome_case,
    load_agent_outcome_eval_cases,
    normalize_eval_text,
    run_agent_outcome_evals,
    summarize_agent_outcome_eval_results,
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
    )

    result = evaluate_agent_outcome_case(case, agent_result)

    assert result["missing_fragments"] == []
    assert result["passed"] is True
    assert result["tool_calls"] == agent_result["tool_calls"]
    assert result["user_input"] == case["user_input"]
    assert result["final_answer"] == agent_result["final_answer"]


def test_evaluate_agent_outcome_case_fails_when_expected_fragment_is_missing() -> None:
    case = AgentOutcomeEvalCase(
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
    )

    result = evaluate_agent_outcome_case(case, agent_result)

    assert result["missing_fragments"] == ["660 грн"]
    assert result["passed"] is False
    assert result["tool_calls"] == agent_result["tool_calls"]
    assert result["user_input"] == case["user_input"]
    assert result["final_answer"] == agent_result["final_answer"]


def test_evaluate_agent_outcome_case_fails_when_forbidden_fragment_is_present() -> None:
    case = AgentOutcomeEvalCase(
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
    )

    result = evaluate_agent_outcome_case(case, agent_result)

    assert result["missing_fragments"] == []
    assert result["unexpected_fragments"] == ["застава не потрібна"]
    assert result["passed"] is False


def test_evaluate_agent_outcome_case_passes_when_required_tool_is_called() -> None:
    case = AgentOutcomeEvalCase(
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
    )

    result = evaluate_agent_outcome_case(case, agent_result)

    assert result["missing_fragments"] == []
    assert result["unexpected_fragments"] == []
    assert result["missing_tool_names"] == []
    assert result["passed"] is True


def test_evaluate_agent_outcome_case_fails_when_required_tool_is_missing() -> None:
    case = AgentOutcomeEvalCase(
        user_input="Знайди 2-місний намет",
        expected_answer_fragments=["2-місн"],
        forbidden_answer_fragments=[],
        required_tool_names=["search_products"],
    )

    agent_result = AgentRunResult(
        final_answer="Можу допомогти знайти 2-місний намет.",
        tool_calls=[],
    )

    result = evaluate_agent_outcome_case(case, agent_result)

    assert result["missing_fragments"] == []
    assert result["unexpected_fragments"] == []
    assert result["missing_tool_names"] == ["search_products"]
    assert result["passed"] is False


def test_evaluate_agent_outcome_case_normalizes_answer_fragments() -> None:
    case = AgentOutcomeEvalCase(
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
    )

    result = evaluate_agent_outcome_case(case, agent_result)

    assert result["missing_fragments"] == []
    assert result["passed"] is True


def test_run_agent_outcome_evals_returns_results_for_all_cases() -> None:
    cases = [
        AgentOutcomeEvalCase(
            user_input="Скільки буде коштувати 2-місний намет на 3 дні?",
            expected_answer_fragments=["660 грн", "2-місн"],
            forbidden_answer_fragments=[],
            required_tool_names=[],
        ),
        AgentOutcomeEvalCase(
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
            )

        return AgentRunResult(
            final_answer="Знайдено туристичний спальник.",
            tool_calls=[
                ToolCall(
                    tool_name="search_products",
                    arguments={"query": "спальник"},
                )
            ],
        )

    results = run_agent_outcome_evals(cases, fake_agent_runner)

    assert len(results) == 2
    assert results[0]["passed"] is True
    assert results[1]["passed"] is True


def test_summarize_agent_outcome_eval_results() -> None:
    results: list[AgentOutcomeEvalResult] = [
        {
            "user_input": "Скільки буде коштувати 2-місний намет на 3 дні?",
            "final_answer": "Оренда 2-місного намету на 3 дні коштує 660 грн.",
            "missing_fragments": [],
            "unexpected_fragments": [],
            "tool_calls": [
                {
                    "tool_name": "search_products",
                    "arguments": {"query": "2-місний намет"},
                }
            ],
            "missing_tool_names": [],
            "passed": True,
        },
        {
            "user_input": "Знайди спальник",
            "final_answer": "Знайдено туристичний спальник.",
            "missing_fragments": [],
            "unexpected_fragments": [],
            "tool_calls": [
                {
                    "tool_name": "search_products",
                    "arguments": {"query": "спальник"},
                }
            ],
            "missing_tool_names": [],
            "passed": True,
        },
        {
            "user_input": "Скільки коштує доставка по Києву?",
            "final_answer": "Доставка коштує 300 грн.",
            "missing_fragments": ["200 грн"],
            "unexpected_fragments": [],
            "tool_calls": [
                {
                    "tool_name": "get_delivery_info",
                    "arguments": {"location": "Київ"},
                }
            ],
            "missing_tool_names": [],
            "passed": False,
        },
    ]

    summary = summarize_agent_outcome_eval_results(results)

    assert summary["total_cases"] == 3
    assert summary["passed_cases"] == 2
    assert summary["pass_rate"] == pytest.approx(2 / 3)


def test_summarize_agent_outcome_eval_results_returns_zero_summary_for_empty_results() -> None:
    results: list[AgentOutcomeEvalResult] = []

    summary = summarize_agent_outcome_eval_results(results)

    assert summary["total_cases"] == 0
    assert summary["passed_cases"] == 0
    assert summary["pass_rate"] == 0.0


def test_load_agent_outcome_eval_cases(tmp_path: Path) -> None:
    cases = [
        AgentOutcomeEvalCase(
            user_input="Скільки буде коштувати 2-місний намет на 3 дні?",
            expected_answer_fragments=[
                "660 грн",
                "2-місн",
            ],
            forbidden_answer_fragments=[],
            required_tool_names=[],
        ),
        AgentOutcomeEvalCase(
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
