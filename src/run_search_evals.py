import datetime

from prokat24_ai_lab.knowledge_base import load_knowledge_base
from prokat24_ai_lab.project_paths import (
    KNOWLEDGE_PAGES_FILE,
    SEARCH_EVAL_CASES_FILE,
    SEARCH_EVAL_REPORT_FILE,
)
from prokat24_ai_lab.search_evals import (
    build_search_eval_report,
    load_search_eval_cases,
    run_search_evals,
    save_search_eval_report,
)


def main() -> None:
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    eval_cases = load_search_eval_cases(SEARCH_EVAL_CASES_FILE)

    search_results = run_search_evals(eval_cases, pages)

    created_at = datetime.datetime.now().isoformat(timespec="seconds")

    report = build_search_eval_report(search_results, created_at)

    save_search_eval_report(report, SEARCH_EVAL_REPORT_FILE)

    print(f"Saved report to {SEARCH_EVAL_REPORT_FILE}")


if __name__ == "__main__":
    main()
