from collections.abc import Sequence

from phoenix.client import Client

from prokat24_ai_lab.evals.agent.report import AgentEvalReport
from prokat24_ai_lab.evals.agent.types import AgentCriterionResult
from prokat24_ai_lab.observability.phoenix_annotations import (
    build_llm_annotation,
    build_outcome_annotation,
)


def build_criteria_explanation(
    criteria: Sequence[AgentCriterionResult],
) -> str:
    return "; ".join(f"{criterion['name']}: {criterion['reason']}" for criterion in criteria)


def publish_agent_eval_annotations(
    client: Client,
    report: AgentEvalReport,
) -> None:
    for result in report["results"]:
        trace_id = result["trace_id"]

        if trace_id is None:
            continue

        outcome_annotation = build_outcome_annotation(
            trace_id=trace_id,
            passed=result["outcome"]["passed"],
        )

        client.traces.add_trace_annotation(
            **outcome_annotation,
            sync=True,
        )

        capability_explanation = build_criteria_explanation(result["capability"]["criteria"])

        capability_annotation = build_llm_annotation(
            trace_id=trace_id,
            annotation_name="capability",
            passed=result["capability"]["passed"],
            explanation=capability_explanation,
        )

        client.traces.add_trace_annotation(
            **capability_annotation,
            sync=True,
        )

        grounding_explanation = build_criteria_explanation(result["grounding"]["criteria"])

        grounding_annotation = build_llm_annotation(
            trace_id=trace_id,
            annotation_name="grounding",
            passed=result["grounding"]["passed"],
            explanation=grounding_explanation,
        )

        client.traces.add_trace_annotation(
            **grounding_annotation,
            sync=True,
        )
