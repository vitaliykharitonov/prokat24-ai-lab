from prokat24_ai_lab.observability.phoenix_annotations import (
    build_llm_annotation,
    build_outcome_annotation,
)


def test_build_outcome_annotation_for_passed_result() -> None:
    annotation = build_outcome_annotation(
        trace_id="abc123",
        passed=True,
    )

    assert annotation == {
        "trace_id": "abc123",
        "annotation_name": "outcome",
        "annotator_kind": "CODE",
        "label": "PASS",
        "score": 1.0,
    }


def test_build_outcome_annotation_for_failed_result() -> None:
    annotation = build_outcome_annotation(
        trace_id="def456",
        passed=False,
    )

    assert annotation == {
        "trace_id": "def456",
        "annotation_name": "outcome",
        "annotator_kind": "CODE",
        "label": "FAIL",
        "score": 0.0,
    }


def test_build_llm_annotation_for_capability() -> None:
    annotation = build_llm_annotation(
        trace_id="abc123",
        annotation_name="capability",
        passed=False,
        explanation="The assistant offered an unsupported action.",
    )

    assert annotation == {
        "trace_id": "abc123",
        "annotation_name": "capability",
        "annotator_kind": "LLM",
        "label": "FAIL",
        "score": 0.0,
        "explanation": "The assistant offered an unsupported action.",
    }


def test_build_llm_annotation_for_grounding() -> None:
    annotation = build_llm_annotation(
        trace_id="abc123",
        annotation_name="grounding",
        passed=False,
        explanation="The answer is supported by tool results.",
    )

    assert annotation == {
        "trace_id": "abc123",
        "annotation_name": "grounding",
        "annotator_kind": "LLM",
        "label": "FAIL",
        "score": 0.0,
        "explanation": "The answer is supported by tool results.",
    }
