from typing import cast
from unittest.mock import MagicMock, call

from prokat24_ai_lab.evals.agent.phoenix import (
    build_criteria_explanation,
    publish_agent_eval_annotations,
)
from prokat24_ai_lab.evals.agent.report import AgentEvalReport


def test_build_criteria_explanation() -> None:
    explanation = build_criteria_explanation(
        [
            {
                "name": "supported_actions_only",
                "passed": False,
                "reason": "The assistant offered an unsupported action.",
            }
        ]
    )

    assert explanation == ("supported_actions_only: The assistant offered an unsupported action.")


def test_publish_agent_eval_annotations() -> None:
    client = MagicMock()

    report = cast(
        AgentEvalReport,
        {
            "results": [
                {
                    "id": "case_1",
                    "trace_id": "abc123",
                    "user_input": "Test question",
                    "final_answer": "Test answer",
                    "tool_calls": [],
                    "tool_executions": [],
                    "outcome": {
                        "passed": True,
                        "missing_fragments": [],
                        "unexpected_fragments": [],
                        "missing_tool_names": [],
                    },
                    "capability": {
                        "passed": False,
                        "failed_criteria": ["supported_actions_only"],
                        "criteria": [
                            {
                                "name": "supported_actions_only",
                                "passed": False,
                                "reason": "Unsupported action.",
                            }
                        ],
                    },
                    "grounding": {
                        "passed": True,
                        "failed_criteria": [],
                        "criteria": [
                            {
                                "name": "grounded_facts_only",
                                "passed": True,
                                "reason": "Grounded.",
                            }
                        ],
                    },
                }
            ]
        },
    )

    publish_agent_eval_annotations(client, report)

    assert client.traces.add_trace_annotation.call_count == 3

    client.traces.add_trace_annotation.assert_has_calls(
        [
            call(
                trace_id="abc123",
                annotation_name="outcome",
                annotator_kind="CODE",
                label="PASS",
                score=1.0,
                sync=True,
            ),
            call(
                trace_id="abc123",
                annotation_name="capability",
                annotator_kind="LLM",
                label="FAIL",
                score=0.0,
                explanation="supported_actions_only: Unsupported action.",
                sync=True,
            ),
            call(
                trace_id="abc123",
                annotation_name="grounding",
                annotator_kind="LLM",
                label="PASS",
                score=1.0,
                explanation="grounded_facts_only: Grounded.",
                sync=True,
            ),
        ]
    )


def test_publish_agent_eval_annotations_skips_result_without_trace_id() -> None:
    client = MagicMock()

    report = cast(
        AgentEvalReport,
        {
            "results": [
                {
                    "id": "case_1",
                    "trace_id": None,
                    "user_input": "Test question",
                    "final_answer": "Test answer",
                    "tool_calls": [],
                    "tool_executions": [],
                    "outcome": {
                        "passed": True,
                        "missing_fragments": [],
                        "unexpected_fragments": [],
                        "missing_tool_names": [],
                    },
                    "capability": {
                        "passed": True,
                        "failed_criteria": [],
                        "criteria": [],
                    },
                    "grounding": {
                        "passed": True,
                        "failed_criteria": [],
                        "criteria": [],
                    },
                }
            ]
        },
    )

    publish_agent_eval_annotations(client, report)

    client.traces.add_trace_annotation.assert_not_called()
