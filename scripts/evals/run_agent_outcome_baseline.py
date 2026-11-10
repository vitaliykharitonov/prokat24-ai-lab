import datetime

from openai import OpenAI
from openinference.instrumentation import suppress_tracing

from prokat24_ai_lab.agents.openai_tool_call import OPENAI_TOOL_CALL_MODEL
from prokat24_ai_lab.evals.agent.capability import evaluate_agent_capability
from prokat24_ai_lab.evals.agent.grounding import evaluate_agent_grounding
from prokat24_ai_lab.evals.agent.outcome import (
    load_agent_outcome_eval_cases,
    run_agent_outcome_evals,
)
from prokat24_ai_lab.evals.agent.presentation import (
    print_agent_eval_results,
    print_agent_eval_summary,
    print_agent_observability,
)
from prokat24_ai_lab.evals.agent.report import (
    AgentEvalConfiguration,
    build_agent_eval_report,
    save_agent_eval_report,
)
from prokat24_ai_lab.evals.agent.runtime import create_agent_eval_runtime
from prokat24_ai_lab.git_metadata import get_current_git_commit, is_git_worktree_dirty
from prokat24_ai_lab.judges.openai_judge import OPENAI_JUDGE_MODEL, create_openai_criterion_judge
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.observability.cost_estimation import (
    OpenAIModelCostRecord,
    OpenAITraceCostRecord,
    aggregate_openai_costs_by_trace,
    estimate_openai_model_cost,
)
from prokat24_ai_lab.observability.phoenix_annotations import (
    publish_agent_eval_annotations,
)
from prokat24_ai_lab.observability.runtime import create_observability_runtime
from prokat24_ai_lab.products.catalog import load_products
from prokat24_ai_lab.project_paths import (
    AGENT_EVAL_REPORTS_DIR,
    AGENT_OUTCOME_EVAL_CASES_FILE,
    KNOWLEDGE_PAGES_FILE,
    PRODUCTS_FILE,
    PROJECT_ROOT,
)

MAX_AGENT_STEPS = 5


def main() -> None:
    cases = load_agent_outcome_eval_cases(AGENT_OUTCOME_EVAL_CASES_FILE)
    products = load_products(PRODUCTS_FILE)
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)

    observability_runtime = create_observability_runtime(
        project_name="prokat24-ai-lab",
    )

    tracer = observability_runtime["tracer"]
    phoenix_client = observability_runtime["phoenix_client"]
    observability_logger = observability_runtime["logger"]

    client = OpenAI()

    criterion_judge = create_openai_criterion_judge(client)

    agent_runtime = create_agent_eval_runtime(
        client,
        products,
        pages,
        tracer=tracer,
        logger=observability_logger,
        max_steps=MAX_AGENT_STEPS,
    )

    agent_runner = agent_runtime["agent_runner"]
    agent_step_records = agent_runtime["agent_step_records"]
    tool_execution_records = agent_runtime["tool_execution_records"]
    model_usage_records = agent_runtime["model_usage_records"]
    eval_trace_ids = agent_runtime["eval_trace_ids"]

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
        publish_agent_eval_annotations(phoenix_client, report)

    historical_file, latest_file = save_agent_eval_report(report, AGENT_EVAL_REPORTS_DIR)

    print_agent_eval_results(results, capability_results, grounding_results)

    print_agent_eval_summary(report["summary"], historical_file, latest_file)

    print_agent_observability(
        tool_execution_records,
        agent_step_records,
        model_usage_records,
        model_cost_records,
        trace_cost_records,
    )


if __name__ == "__main__":
    main()
