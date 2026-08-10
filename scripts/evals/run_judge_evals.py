from openai import OpenAI

from prokat24_ai_lab.evals.semantic_grading import (
    evaluate_judge_case,
    load_judge_eval_cases,
)
from prokat24_ai_lab.judges.openai_judge import create_openai_judge
from prokat24_ai_lab.project_paths import JUDGE_EVAL_CASES_FILE

RUNS = 3


def main() -> None:
    cases = load_judge_eval_cases(JUDGE_EVAL_CASES_FILE)
    client = OpenAI()
    judge = create_openai_judge(client)

    stable_cases = 0
    matched_cases = 0

    for index, case in enumerate(cases, start=1):
        case_results = []

        for _ in range(RUNS):
            result = evaluate_judge_case(case, judge)
            case_results.append(result)

        passed_values = [result["actual_passed"] for result in case_results]
        scores = [result["score"] for result in case_results]
        stability = len(set(passed_values)) == 1
        score_stability = len(set(scores)) == 1
        matched_values = [result["matched"] for result in case_results]
        all_matched = all(matched_values)

        if stability:
            stable_cases += 1

        if all_matched:
            matched_cases += 1

        print(f"\nCase {index}")
        print(f"Expected: {case['expected_passed']}")
        print(f"Actual: {passed_values}")
        print(f"Scores: {scores}")
        print(f"Stability: {stability}")
        print(f"Score stability: {score_stability}")
        print(f"All matched: {all_matched}")

    print("\nSummary")
    print(f"Stable cases: {stable_cases}/{len(cases)}")
    print(f"All matched:  {matched_cases}/{len(cases)}")


if __name__ == "__main__":
    main()
