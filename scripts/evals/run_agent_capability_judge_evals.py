from openai import OpenAI

from prokat24_ai_lab.evals.agent_evals import (
    load_agent_capability_judge_cases,
    run_agent_capability_judge_evals,
    summarize_agent_capability_judge_results,
)
from prokat24_ai_lab.judges.openai_judge import create_openai_criterion_judge
from prokat24_ai_lab.project_paths import AGENT_CAPABILITY_JUDGE_CASES_FILE


def main() -> None:
    cases = load_agent_capability_judge_cases(AGENT_CAPABILITY_JUDGE_CASES_FILE)

    client = OpenAI()
    judge = create_openai_criterion_judge(client)

    results = run_agent_capability_judge_evals(
        cases,
        judge,
    )

    summary = summarize_agent_capability_judge_results(results)

    for result in results:
        print(f"ID: {result['id']}")
        print(f"Category: {result['category']}")
        print(f"Expected passed: {result['expected_passed']}")
        print(f"Actual passed: {result['actual_passed']}")
        print(f"Matched: {result['matched']}")

        for criterion in result["criteria"]:
            print(f"Criterion: {criterion['name']} | Passed: {criterion['passed']}")
            print(f"Reason: {criterion['reason']}")

        print()

    print("MISMATCHES\n")

    mismatches = [result for result in results if not result["matched"]]

    if not mismatches:
        print("None")
    else:
        for result in mismatches:
            print(f"ID: {result['id']}")
            print(f"Category: {result['category']}")
            print(f"Expected passed: {result['expected_passed']}")
            print(f"Actual passed: {result['actual_passed']}")

            for criterion in result["criteria"]:
                print(f"Reason: {criterion['reason']}")

            print()

    print("AGENT CAPABILITY JUDGE SUMMARY\n")
    print(f"Cases: {summary['total_cases']}")
    print(f"Matches: {summary['matched_cases']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")


if __name__ == "__main__":
    main()
