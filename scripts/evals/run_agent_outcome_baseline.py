import datetime
import logging
import os
from uuid import uuid4

from openai import OpenAI
from openinference.instrumentation import suppress_tracing
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.trace import SpanContext
from phoenix.client import Client
from phoenix.otel import register

from prokat24_ai_lab.agents.agent_loop import run_agent
from prokat24_ai_lab.agents.openai_agent_step import create_openai_agent_step_generator
from prokat24_ai_lab.agents.openai_tool_call import OPENAI_TOOL_CALL_MODEL
from prokat24_ai_lab.agents.openai_tool_schema import create_openai_tools
from prokat24_ai_lab.agents.tool_executor import create_tool_executor
from prokat24_ai_lab.agents.tool_registry import create_tool_registry
from prokat24_ai_lab.agents.types import AgentRunResult
from prokat24_ai_lab.evals.agent.capability import evaluate_agent_capability
from prokat24_ai_lab.evals.agent.grounding import evaluate_agent_grounding
from prokat24_ai_lab.evals.agent.outcome import (
    load_agent_outcome_eval_cases,
    run_agent_outcome_evals,
)
from prokat24_ai_lab.evals.agent.report import (
    AgentEvalConfiguration,
    build_agent_eval_report,
    save_agent_eval_report,
)
from prokat24_ai_lab.git_metadata import get_current_git_commit, is_git_worktree_dirty
from prokat24_ai_lab.judges.openai_judge import OPENAI_JUDGE_MODEL, create_openai_criterion_judge
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.observability.agent_step_instrumentation import (
    AgentStepExecutionRecord,
    create_instrumented_agent_step_generator,
)
from prokat24_ai_lab.observability.context import (
    current_eval_case_id,
    current_sequence_number,
    current_trace_id,
)
from prokat24_ai_lab.observability.cost_estimation import (
    OpenAIModelCostRecord,
    OpenAITraceCostRecord,
    aggregate_openai_costs_by_trace,
    estimate_openai_model_cost,
)
from prokat24_ai_lab.observability.openai_usage import OpenAIModelUsageRecord
from prokat24_ai_lab.observability.otel_instrumentation import (
    create_traced_agent_step_generator,
    create_traced_tool_executor,
)
from prokat24_ai_lab.observability.phoenix_annotations import (
    build_criteria_explanation,
    build_llm_annotation,
    build_outcome_annotation,
)
from prokat24_ai_lab.observability.structured_logging import (
    build_agent_step_log_event,
    build_model_usage_log_event,
    build_tool_execution_log_event,
    log_observability_event,
)
from prokat24_ai_lab.observability.tool_instrumentation import (
    ToolExecutionRecord,
    create_instrumented_tool_executor,
)
from prokat24_ai_lab.products.catalog import load_products
from prokat24_ai_lab.project_paths import (
    AGENT_EVAL_REPORTS_DIR,
    AGENT_OUTCOME_EVAL_CASES_FILE,
    KNOWLEDGE_PAGES_FILE,
    PRODUCTS_FILE,
    PROJECT_ROOT,
)

MAX_AGENT_STEPS = 5


def get_phoenix_base_url() -> str | None:
    return os.getenv("PHOENIX_BASE_URL")


def get_trace_id(span_context: SpanContext) -> str | None:
    if not span_context.is_valid:
        return None

    return format(span_context.trace_id, "032x")


def get_correlation_id(trace_id: str | None) -> str:
    if trace_id is not None:
        return trace_id

    return uuid4().hex


def main() -> None:
    cases = load_agent_outcome_eval_cases(AGENT_OUTCOME_EVAL_CASES_FILE)
    products = load_products(PRODUCTS_FILE)
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)

    phoenix_base_url = get_phoenix_base_url()

    if phoenix_base_url is not None:
        tracer_provider = register(
            project_name="prokat24-ai-lab",
            endpoint=f"{phoenix_base_url.rstrip('/')}/v1/traces",
            auto_instrument=False,
        )

        OpenAIInstrumentor().instrument(
            tracer_provider=tracer_provider,
        )

        tracer = tracer_provider.get_tracer(__name__)
        phoenix_client: Client | None = Client(
            base_url=phoenix_base_url,
        )

    else:
        tracer = trace.get_tracer(__name__)
        phoenix_client = None

    client = OpenAI()

    criterion_judge = create_openai_criterion_judge(client)

    registry = create_tool_registry(products, pages)
    openai_tools = create_openai_tools(registry)

    observability_logger = logging.getLogger("prokat24_ai_lab.observability")
    observability_logger.setLevel(logging.INFO)

    observability_handler = logging.StreamHandler()
    observability_handler.setFormatter(logging.Formatter("%(message)s"))

    observability_logger.addHandler(observability_handler)
    observability_logger.propagate = False

    def on_agent_step_record(record: AgentStepExecutionRecord) -> None:
        agent_step_records.append(record)

        log_event = build_agent_step_log_event(record)
        log_observability_event(observability_logger, log_event)

    def on_model_usage_record(record: OpenAIModelUsageRecord) -> None:
        model_usage_records.append(record)

        log_event = build_model_usage_log_event(record)
        log_observability_event(observability_logger, log_event)

    def on_tool_execution_record(record: ToolExecutionRecord) -> None:
        tool_execution_records.append(record)

        log_event = build_tool_execution_log_event(record)
        log_observability_event(observability_logger, log_event)

    agent_step_records: list[AgentStepExecutionRecord] = []
    model_usage_records: list[OpenAIModelUsageRecord] = []
    eval_trace_ids: dict[str, str] = {}

    base_agent_step_generator = create_openai_agent_step_generator(
        client,
        openai_tools,
        on_usage=on_model_usage_record,
    )

    traced_agent_step_generator = create_traced_agent_step_generator(
        generator=base_agent_step_generator,
        tracer=tracer,
    )

    instrumented_agent_step_generator = create_instrumented_agent_step_generator(
        generator=traced_agent_step_generator,
        on_record=on_agent_step_record,
    )

    tool_execution_records: list[ToolExecutionRecord] = []

    base_tool_executor = create_tool_executor(registry)

    traced_tool_executor = create_traced_tool_executor(
        executor=base_tool_executor,
        tracer=tracer,
    )

    instrumented_executor = create_instrumented_tool_executor(
        executor=traced_tool_executor,
        on_record=on_tool_execution_record,
    )

    def agent_runner(user_input: str) -> AgentRunResult:
        case_id = current_eval_case_id.get()
        sequence_token = current_sequence_number.set(0)

        try:
            with tracer.start_as_current_span("agent.run") as span:
                span.set_attribute("openinference.span.kind", "AGENT")

                trace_id = get_trace_id(span.get_span_context())
                correlation_id = get_correlation_id(trace_id)

                trace_token = current_trace_id.set(correlation_id)

                try:
                    if case_id is not None:
                        span.set_attribute("eval.case_id", case_id)

                        if trace_id is not None:
                            eval_trace_ids[case_id] = trace_id

                    span.set_attribute("input.value", user_input)
                    span.set_attribute("input.mime_type", "text/plain")

                    agent_result = run_agent(
                        user_input,
                        instrumented_agent_step_generator,
                        instrumented_executor,
                        max_steps=MAX_AGENT_STEPS,
                    )

                    span.set_attribute("output.value", agent_result["final_answer"])
                    span.set_attribute("output.mime_type", "text/plain")

                    return agent_result

                finally:
                    current_trace_id.reset(trace_token)

        finally:
            current_sequence_number.reset(sequence_token)

    results = run_agent_outcome_evals(cases, agent_runner)

    for result in results:
        result["trace_id"] = eval_trace_ids.get(result["id"])

    model_cost_records: list[OpenAIModelCostRecord] = [
        estimate_openai_model_cost(record) for record in model_usage_records
    ]

    trace_cost_records: list[OpenAITraceCostRecord] = aggregate_openai_costs_by_trace(
        model_cost_records
    )

    with suppress_tracing():
        capability_results = []

        for result in results:
            capability_result = evaluate_agent_capability(result, criterion_judge)
            capability_results.append(capability_result)

        grounding_results = []

        for result in results:
            grounding_result = evaluate_agent_grounding(result, criterion_judge)
            grounding_results.append(grounding_result)

    run_time = datetime.datetime.now(datetime.UTC)

    created_at = run_time.isoformat(timespec="seconds")
    run_timestamp = run_time.strftime("%Y%m%dT%H%M%SZ")

    git_commit = get_current_git_commit()
    git_dirty = is_git_worktree_dirty()

    short_commit = git_commit[:7] if git_commit is not None else "no-git"
    run_id = f"{run_timestamp}_{short_commit}"

    configuration = AgentEvalConfiguration(
        agent_model=OPENAI_TOOL_CALL_MODEL,
        judge_model=OPENAI_JUDGE_MODEL,
        max_agent_steps=MAX_AGENT_STEPS,
        eval_cases_file=AGENT_OUTCOME_EVAL_CASES_FILE.relative_to(PROJECT_ROOT).as_posix(),
    )

    report = build_agent_eval_report(
        results,
        capability_results,
        grounding_results,
        configuration=configuration,
        run_id=run_id,
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    if phoenix_client is not None:
        for result in report["results"]:
            trace_id = result["trace_id"]

            if trace_id is None:
                continue

            outcome_annotation = build_outcome_annotation(
                trace_id=trace_id,
                passed=result["outcome"]["passed"],
            )

            phoenix_client.traces.add_trace_annotation(
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

            phoenix_client.traces.add_trace_annotation(
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

            phoenix_client.traces.add_trace_annotation(
                **grounding_annotation,
                sync=True,
            )

    summary = report["summary"]

    historical_file, latest_file = save_agent_eval_report(report, AGENT_EVAL_REPORTS_DIR)

    for result, capability_result, grounding_result in zip(
        results,
        capability_results,
        grounding_results,
        strict=True,
    ):
        print("=" * 80)
        print(f"Question: {result['user_input']}")
        print()

        print("OUTCOME")
        print(f"Passed: {result['passed']}")
        print(f"Missing fragments: {result['missing_fragments']}")
        print(f"Unexpected fragments: {result['unexpected_fragments']}")
        print(f"Missing tool names: {result['missing_tool_names']}")
        print()

        print("CAPABILITY")
        print(f"Passed: {capability_result['passed']}")
        print(f"Failed criteria: {capability_result['failed_criteria']}")

        for criterion in capability_result["criteria"]:
            print(f"- {criterion['name']}: {'PASS' if criterion['passed'] else 'FAIL'}")
            print(f"  Reason: {criterion['reason']}")

        print()

        print("GROUNDING")
        print(f"Passed: {grounding_result['passed']}")
        print(f"Failed criteria: {grounding_result['failed_criteria']}")

        for criterion in grounding_result["criteria"]:
            print(f"- {criterion['name']}: {'PASS' if criterion['passed'] else 'FAIL'}")
            print(f"  Reason: {criterion['reason']}")

        print()

        print("ANSWER")
        print(result["final_answer"])
        print()

        print("TOOL CALLS")
        print(result["tool_calls"])
        print()

        print("TOOL EXECUTIONS")
        print(result["tool_executions"])
        print()

    print("=" * 80)
    print("SUMMARY")
    print()

    print("AGENT OUTCOME")
    print(f"Cases: {summary['outcome']['total_cases']}")
    print(f"Passed: {summary['outcome']['passed_cases']}")
    print(f"Pass rate: {summary['outcome']['pass_rate']:.2%}")
    print()

    print("AGENT CAPABILITY")
    print(f"Cases: {summary['capability']['total_cases']}")
    print(f"Passed: {summary['capability']['passed_cases']}")
    print(f"Pass rate: {summary['capability']['pass_rate']:.2%}")
    print()

    print("AGENT GROUNDING")
    print(f"Cases: {summary['grounding']['total_cases']}")
    print(f"Passed: {summary['grounding']['passed_cases']}")
    print(f"Pass rate: {summary['grounding']['pass_rate']:.2%}")

    print()
    print(f"Historical report saved to {historical_file}")
    print(f"Latest report saved to {latest_file}")

    print()
    print("=" * 80)
    print("TOOL EXECUTIONS")
    print()

    for record in tool_execution_records:
        print(
            f"trace={record['trace_id']} | "
            f"seq={record['sequence_number']} | "
            f"started={record['started_at']} | "
            f"tool={record['tool_name']} | "
            f"{record['duration_ms']:.2f} ms | "
            f"{record['status']}"
        )

    print()
    print("=" * 80)
    print("AGENT STEPS")
    print()

    for record in agent_step_records:
        print(
            f"trace={record['trace_id']} | "
            f"seq={record['sequence_number']} | "
            f"step={record['step_number']} | "
            f"started={record['started_at']} | "
            f"decision={record['decision']} | "
            f"{record['duration_ms']:.2f} ms | "
            f"{record['status']}"
        )

    print()
    print("=" * 80)
    print("MODEL USAGE")
    print()

    for record in model_usage_records:
        print(
            f"trace={record['trace_id']} | "
            f"step={record['step_number']} | "
            f"model={record['model']} | "
            f"input={record['input_tokens']} | "
            f"cached={record['cached_input_tokens']} | "
            f"output={record['output_tokens']} | "
            f"reasoning={record['reasoning_tokens']} | "
            f"total={record['total_tokens']}"
        )

    print()
    print("=" * 80)
    print("MODEL COST")
    print()

    for record in model_cost_records:
        print(
            f"trace={record['trace_id']} | "
            f"step={record['step_number']} | "
            f"model={record['model']} | "
            f"input=${record['input_cost_usd']:.6f} | "
            f"cached=${record['cached_input_cost_usd']:.6f} | "
            f"output=${record['output_cost_usd']:.6f} | "
            f"total=${record['total_cost_usd']:.6f}"
        )

    print()
    print("=" * 80)
    print("TRACE COST")
    print()

    for record in trace_cost_records:
        print(
            f"trace={record['trace_id']} | "
            f"calls={record['model_call_count']} | "
            f"input=${record['input_cost_usd']:.6f} | "
            f"cached=${record['cached_input_cost_usd']:.6f} | "
            f"output=${record['output_cost_usd']:.6f} | "
            f"total=${record['total_cost_usd']:.6f}"
        )

    total_agent_cost_usd = sum(record["total_cost_usd"] for record in trace_cost_records)

    average_trace_cost_usd = (
        total_agent_cost_usd / len(trace_cost_records) if trace_cost_records else 0.0
    )

    print()
    print(f"Total agent model cost: ${total_agent_cost_usd:.6f}")
    print(f"Average cost per trace: ${average_trace_cost_usd:.6f}")


if __name__ == "__main__":
    main()
