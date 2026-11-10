import logging
import os
from typing import TypedDict, cast

from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.trace import Tracer
from phoenix.client import Client
from phoenix.otel import register


class ObservabilityRuntime(TypedDict):
    tracer: Tracer
    phoenix_client: Client | None
    logger: logging.Logger


def create_observability_runtime(
    project_name: str,
) -> ObservabilityRuntime:
    phoenix_base_url = os.getenv("PHOENIX_BASE_URL")

    tracer: Tracer

    if phoenix_base_url is not None:
        tracer_provider = register(
            project_name=project_name,
            endpoint=f"{phoenix_base_url.rstrip('/')}/v1/traces",
            auto_instrument=False,
        )

        OpenAIInstrumentor().instrument(
            tracer_provider=tracer_provider,
        )

        tracer = cast(Tracer, tracer_provider.get_tracer(__name__))

        phoenix_client: Client | None = Client(
            base_url=phoenix_base_url,
        )

    else:
        tracer = trace.get_tracer(__name__)
        phoenix_client = None

    observability_logger = logging.getLogger("prokat24_ai_lab.observability")
    observability_logger.setLevel(logging.INFO)

    observability_handler = logging.StreamHandler()
    observability_handler.setFormatter(logging.Formatter("%(message)s"))

    observability_logger.addHandler(observability_handler)
    observability_logger.propagate = False

    return ObservabilityRuntime(
        tracer=tracer,
        phoenix_client=phoenix_client,
        logger=observability_logger,
    )
