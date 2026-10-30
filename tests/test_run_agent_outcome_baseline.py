from opentelemetry.trace import INVALID_SPAN_CONTEXT, SpanContext, TraceFlags
from scripts.evals.run_agent_outcome_baseline import (
    get_correlation_id,
    get_phoenix_base_url,
    get_trace_id,
)


def test_get_phoenix_base_url_returns_none_when_not_configured(
    monkeypatch,
) -> None:
    monkeypatch.delenv("PHOENIX_BASE_URL", raising=False)

    assert get_phoenix_base_url() is None


def test_get_phoenix_base_url_returns_configured_url(
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "PHOENIX_BASE_URL",
        "http://localhost:6006",
    )

    assert get_phoenix_base_url() == "http://localhost:6006"


def test_get_trace_id_returns_none_for_invalid_span_context() -> None:
    assert get_trace_id(INVALID_SPAN_CONTEXT) is None


def test_get_trace_id_returns_formatted_id_for_valid_span_context() -> None:
    span_context = SpanContext(
        trace_id=int("1234567890abcdef1234567890abcdef", 16),
        span_id=1,
        is_remote=False,
        trace_flags=TraceFlags(0),
        trace_state=None,
    )

    assert get_trace_id(span_context) == "1234567890abcdef1234567890abcdef"


def test_get_correlation_id_reuses_trace_id() -> None:
    assert get_correlation_id("trace-123") == "trace-123"


def test_get_correlation_id_generates_id_when_trace_id_is_missing() -> None:
    correlation_id = get_correlation_id(None)

    assert len(correlation_id) == 32
    int(correlation_id, 16)
