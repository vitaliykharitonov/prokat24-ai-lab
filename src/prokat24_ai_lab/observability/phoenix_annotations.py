from typing import Literal


def build_outcome_annotation(
    trace_id: str,
    passed: bool,
) -> dict[str, str | float]:
    return {
        "trace_id": trace_id,
        "annotation_name": "outcome",
        "annotator_kind": "CODE",
        "label": "PASS" if passed else "FAIL",
        "score": 1.0 if passed else 0.0,
    }


def build_llm_annotation(
    trace_id: str,
    annotation_name: Literal["capability", "grounding"],
    passed: bool,
    explanation: str,
) -> dict[str, str | float]:
    return {
        "trace_id": trace_id,
        "annotation_name": annotation_name,
        "annotator_kind": "LLM",
        "label": "PASS" if passed else "FAIL",
        "score": 1.0 if passed else 0.0,
        "explanation": explanation,
    }


def build_criteria_explanation(
    criteria: list[dict[str, str | bool]],
) -> str:
    return "; ".join(f"{criterion['name']}: {criterion['reason']}" for criterion in criteria)
