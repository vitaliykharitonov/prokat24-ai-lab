from openai import OpenAI

from prokat24_ai_lab.evals.semantic_grading import (
    build_grader_confusion_matrix,
    load_grader_comparison_cases,
    run_grader_comparison_evals,
    summarize_grader_comparison,
    summarize_grader_comparison_by_category,
)
from prokat24_ai_lab.judges.openai_judge import create_openai_judge
from prokat24_ai_lab.project_paths import GRADER_COMPARISON_CASES_FILE


def main() -> None:
    cases = load_grader_comparison_cases(GRADER_COMPARISON_CASES_FILE)
    client = OpenAI()

    judge = create_openai_judge(client)

    results = run_grader_comparison_evals(cases, judge)

    summary = summarize_grader_comparison(results)
    summaries = summarize_grader_comparison_by_category(results)

    total_cases = summary["total_cases"]

    deterministic_matrix = build_grader_confusion_matrix(
        results,
        "deterministic_passed",
    )

    llm_matrix = build_grader_confusion_matrix(
        results,
        "llm_passed",
    )

    print("\nGRADER COMPARISON\n")
    print(f"Cases: {total_cases}\n")
    print("Overall:")
    print(
        f"Deterministic: {summary['deterministic_matched']}/{total_cases} "
        f"({summary['deterministic_accuracy']:.2%})"
    )
    print(f"LLM judge: {summary['llm_matched']}/{total_cases} ({summary['llm_accuracy']:.2%})")
    print("\nBy category:\n")

    for category in summaries:
        category_total = summaries[category]["total_cases"]
        print(f"{category}:")
        print(
            f"Deterministic: {summaries[category]['deterministic_matched']}/{category_total} "
            f"({summaries[category]['deterministic_accuracy']:.2%})"
        )
        print(
            f"LLM judge: {summaries[category]['llm_matched']}/{category_total} "
            f"({summaries[category]['llm_accuracy']:.2%})"
        )
        print()

    print("Mismatches:\n")

    for result in results:
        if not result["deterministic_matched"]:
            print(f"ID: {result['id']} [{result['category']}]")
            print(f"Expected: {result['expected_passed']}")
            print(f"Deterministic: {result['deterministic_passed']}")
            print()

        if not result["llm_matched"]:
            print(f"ID: {result['id']} [{result['category']}]")
            print(f"Expected: {result['expected_passed']}")
            print(f"LLM judge: {result['llm_passed']}")
            print(f"Score: {result['llm_score']}")
            print(f"Reason: {result['llm_reason']}")
            print(f"Failed criteria: {result['llm_failed_criteria']}")
            print()

    print("Confusion matrices:\n")
    print("Deterministic:")
    print("                 Human")
    print("             PASS     FAIL")
    print(
        f"Grader PASS  {deterministic_matrix['true_positive']:>4} "
        f"{deterministic_matrix['false_positive']:>8}"
    )
    print(
        f"Grader FAIL  {deterministic_matrix['false_negative']:>4} "
        f"{deterministic_matrix['true_negative']:>8}"
    )
    print()
    print("LLM judge:")
    print("                 Human")
    print("             PASS     FAIL")
    print(f"Grader PASS  {llm_matrix['true_positive']:>4} {llm_matrix['false_positive']:>8}")
    print(f"Grader FAIL  {llm_matrix['false_negative']:>4} {llm_matrix['true_negative']:>8}")


if __name__ == "__main__":
    main()
