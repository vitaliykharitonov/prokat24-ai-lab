from prokat24_ai_lab.knowledge_base import load_knowledge_base
from prokat24_ai_lab.project_paths import KNOWLEDGE_PAGES_FILE, SEARCH_EVAL_CASES_FILE
from prokat24_ai_lab.search_evals import (
    calculate_accuracy,
    load_search_eval_cases,
    run_search_evals,
)


def main() -> None:
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    eval_cases = load_search_eval_cases(SEARCH_EVAL_CASES_FILE)

    search_results = run_search_evals(eval_cases, pages)

    print("Search eval results:")

    for search_result in search_results:
        status = "PASSED" if search_result["passed"] else "FAILED"

        print()
        print(f"{status} | {search_result['query']}")
        print(f"Expected: {search_result['expected_url']}")
        print(f"Actual: {search_result['actual_url']}")

    passed_count = 0

    for result in search_results:
        if result["passed"]:
            passed_count += 1

    accuracy = calculate_accuracy(search_results)
    accuracy_percentage = accuracy * 100

    print()
    print(f"Accuracy: {passed_count}/{len(search_results)} = {accuracy_percentage:.2f}%")


if __name__ == "__main__":
    main()