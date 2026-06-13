import datetime

from prokat24_ai_lab.answer_evals import (
    build_answer_eval_report,
    load_answer_eval_cases,
    run_answer_evals,
    save_answer_eval_report,
)
from prokat24_ai_lab.knowledge_base import load_knowledge_base
from prokat24_ai_lab.project_paths import (
    ANSWER_EVAL_CASES_FILE,
    ANSWER_EVAL_REPORT_FILE,
    KNOWLEDGE_PAGES_FILE,
)


def main() -> None:
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    eval_cases = load_answer_eval_cases(ANSWER_EVAL_CASES_FILE)

    answer_results = run_answer_evals(eval_cases, pages)

    created_at = datetime.datetime.now().isoformat(timespec="seconds")

    report = build_answer_eval_report(answer_results, created_at)

    save_answer_eval_report(report, ANSWER_EVAL_REPORT_FILE)

    print(f"Saved report to {ANSWER_EVAL_REPORT_FILE}")


if __name__ == "__main__":
    main()
