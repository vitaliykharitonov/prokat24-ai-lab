import logging
from unittest.mock import MagicMock

from pytest import MonkeyPatch

from prokat24_ai_lab.observability import runtime


def test_create_observability_runtime_without_phoenix(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.delenv("PHOENIX_BASE_URL", raising=False)

    tracer = MagicMock()
    monkeypatch.setattr(runtime.trace, "get_tracer", MagicMock(return_value=tracer))

    result = runtime.create_observability_runtime(
        project_name="prokat24-ai-lab",
    )

    assert result["tracer"] is tracer
    assert result["phoenix_client"] is None

    logger = result["logger"]

    assert isinstance(logger, logging.Logger)
    assert logger.name == "prokat24_ai_lab.observability"
    assert logger.level == logging.INFO
    assert logger.propagate is False


def test_create_observability_runtime_with_phoenix(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "PHOENIX_BASE_URL",
        "http://localhost:6006",
    )

    tracer = MagicMock()
    tracer_provider = MagicMock()
    tracer_provider.get_tracer.return_value = tracer

    register_mock = MagicMock(return_value=tracer_provider)
    monkeypatch.setattr(runtime, "register", register_mock)

    instrumentor = MagicMock()
    instrumentor_class_mock = MagicMock(return_value=instrumentor)
    monkeypatch.setattr(
        runtime,
        "OpenAIInstrumentor",
        instrumentor_class_mock,
    )

    phoenix_client = MagicMock()
    client_class_mock = MagicMock(return_value=phoenix_client)
    monkeypatch.setattr(
        runtime,
        "Client",
        client_class_mock,
    )

    result = runtime.create_observability_runtime(
        project_name="prokat24-ai-lab",
    )

    register_mock.assert_called_once_with(
        project_name="prokat24-ai-lab",
        endpoint="http://localhost:6006/v1/traces",
        auto_instrument=False,
    )

    instrumentor_class_mock.assert_called_once_with()
    instrumentor.instrument.assert_called_once_with(
        tracer_provider=tracer_provider,
    )

    tracer_provider.get_tracer.assert_called_once()

    client_class_mock.assert_called_once_with(
        base_url="http://localhost:6006",
    )

    assert result["tracer"] is tracer
    assert result["phoenix_client"] is phoenix_client

    logger = result["logger"]

    assert isinstance(logger, logging.Logger)
    assert logger.name == "prokat24_ai_lab.observability"
    assert logger.level == logging.INFO
    assert logger.propagate is False
