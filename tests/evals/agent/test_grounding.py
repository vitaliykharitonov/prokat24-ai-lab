from prokat24_ai_lab.agents.types import ToolExecution
from prokat24_ai_lab.evals.agent.grounding import (
    build_agent_grounding_context,
    evaluate_agent_grounding,
)
from prokat24_ai_lab.evals.agent.types import AgentOutcomeEvalResult
from prokat24_ai_lab.evals.grading.judge import (
    CriterionJudgeResult,
    CriterionResult,
    JudgeInput,
    Rubric,
)


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
