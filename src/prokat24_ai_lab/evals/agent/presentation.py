from pathlib import Path

from prokat24_ai_lab.evals.agent.grounding import AgentGroundingEvalResult
from prokat24_ai_lab.evals.agent.report import AgentEvalSummary
from prokat24_ai_lab.evals.agent.types import AgentCriterionEvalResult, AgentOutcomeEvalResult
from prokat24_ai_lab.observability.agent_step_instrumentation import AgentStepExecutionRecord
from prokat24_ai_lab.observability.cost_estimation import (
    OpenAIModelCostRecord,
    OpenAITraceCostRecord,
)
from prokat24_ai_lab.observability.openai_usage import OpenAIModelUsageRecord
from prokat24_ai_lab.observability.tool_instrumentation import ToolExecutionRecord


def print_agent_eval_results(
    results: list[AgentOutcomeEvalResult],
    capability_results: list[AgentCriterionEvalResult],
    grounding_results: list[AgentGroundingEvalResult],
) -> None:
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


def print_agent_eval_summary(
    summary: AgentEvalSummary,
    historical_file: Path,
    latest_file: Path,
) -> None:
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


def print_agent_observability(
    tool_execution_records: list[ToolExecutionRecord],
    agent_step_records: list[AgentStepExecutionRecord],
    model_usage_records: list[OpenAIModelUsageRecord],
    model_cost_records: list[OpenAIModelCostRecord],
    trace_cost_records: list[OpenAITraceCostRecord],
) -> None:
    print()
    print("=" * 80)
    print("TOOL EXECUTIONS")
    print()

    for tool_record in tool_execution_records:
        print(
            f"trace={tool_record['trace_id']} | "
            f"seq={tool_record['sequence_number']} | "
            f"started={tool_record['started_at']} | "
            f"tool={tool_record['tool_name']} | "
            f"{tool_record['duration_ms']:.2f} ms | "
            f"{tool_record['status']}"
        )

    print()
    print("=" * 80)
    print("AGENT STEPS")
    print()

    for step_record in agent_step_records:
        print(
            f"trace={step_record['trace_id']} | "
            f"seq={step_record['sequence_number']} | "
            f"step={step_record['step_number']} | "
            f"started={step_record['started_at']} | "
            f"decision={step_record['decision']} | "
            f"{step_record['duration_ms']:.2f} ms | "
            f"{step_record['status']}"
        )

    print()
    print("=" * 80)
    print("MODEL USAGE")
    print()

    for usage_record in model_usage_records:
        print(
            f"trace={usage_record['trace_id']} | "
            f"step={usage_record['step_number']} | "
            f"model={usage_record['model']} | "
            f"input={usage_record['input_tokens']} | "
            f"cached={usage_record['cached_input_tokens']} | "
            f"output={usage_record['output_tokens']} | "
            f"reasoning={usage_record['reasoning_tokens']} | "
            f"total={usage_record['total_tokens']}"
        )

    print()
    print("=" * 80)
    print("MODEL COST")
    print()

    for model_cost_record in model_cost_records:
        print(
            f"trace={model_cost_record['trace_id']} | "
            f"step={model_cost_record['step_number']} | "
            f"model={model_cost_record['model']} | "
            f"input=${model_cost_record['input_cost_usd']:.6f} | "
            f"cached=${model_cost_record['cached_input_cost_usd']:.6f} | "
            f"output=${model_cost_record['output_cost_usd']:.6f} | "
            f"total=${model_cost_record['total_cost_usd']:.6f}"
        )

    print()
    print("=" * 80)
    print("TRACE COST")
    print()

    for trace_cost_record in trace_cost_records:
        print(
            f"trace={trace_cost_record['trace_id']} | "
            f"calls={trace_cost_record['model_call_count']} | "
            f"input=${trace_cost_record['input_cost_usd']:.6f} | "
            f"cached=${trace_cost_record['cached_input_cost_usd']:.6f} | "
            f"output=${trace_cost_record['output_cost_usd']:.6f} | "
            f"total=${trace_cost_record['total_cost_usd']:.6f}"
        )

    total_agent_cost_usd = sum(
        trace_cost_record["total_cost_usd"] for trace_cost_record in trace_cost_records
    )

    average_trace_cost_usd = (
        total_agent_cost_usd / len(trace_cost_records) if trace_cost_records else 0.0
    )

    print()
    print(f"Total agent model cost: ${total_agent_cost_usd:.6f}")
    print(f"Average cost per trace: ${average_trace_cost_usd:.6f}")
