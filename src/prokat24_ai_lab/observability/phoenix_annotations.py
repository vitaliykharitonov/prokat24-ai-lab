from typing import Literal, TypedDict


class OutcomeAnnotation(TypedDict):
    trace_id: str
    annotation_name: Literal["outcome"]
    annotator_kind: Literal["CODE"]
    label: str
    score: float


class LLMAnnotation(TypedDict):
    trace_id: str
    annotation_name: Literal["capability", "grounding"]
    annotator_kind: Literal["LLM"]
    label: str
    score: float
    explanation: str


def build_outcome_annotation(
    trace_id: str,
    passed: bool,
) -> OutcomeAnnotation:
    return OutcomeAnnotation(
        trace_id=trace_id,
        annotation_name="outcome",
        annotator_kind="CODE",
        label="PASS" if passed else "FAIL",
        score=1.0 if passed else 0.0,
    )


def build_llm_annotation(
    trace_id: str,
    annotation_name: Literal["capability", "grounding"],
    passed: bool,
    explanation: str,
) -> LLMAnnotation:
    return LLMAnnotation(
        trace_id=trace_id,
        annotation_name=annotation_name,
        annotator_kind="LLM",
        label="PASS" if passed else "FAIL",
        score=1.0 if passed else 0.0,
        explanation=explanation,
    )
