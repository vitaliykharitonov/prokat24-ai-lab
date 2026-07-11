import datetime

from prokat24_ai_lab.eval_exit_codes import get_eval_exit_code
from prokat24_ai_lab.git_metadata import get_current_git_commit, is_git_worktree_dirty
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


def main() -> int:
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    eval_cases = load_search_eval_cases(SEARCH_EVAL_CASES_FILE)

    search_results = run_search_evals(eval_cases, pages)

    created_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    git_commit = get_current_git_commit()
    git_dirty = is_git_worktree_dirty()

    report = build_search_eval_report(
        search_results,
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    print(f"Search evals: {report['passed_cases']}/{report['total_cases']} passed")

    save_search_eval_report(report, SEARCH_EVAL_REPORT_FILE)

    print(f"Saved report to {SEARCH_EVAL_REPORT_FILE}")

    failed_cases = report["total_cases"] - report["passed_cases"]

    return get_eval_exit_code(failed_cases)


if __name__ == "__main__":
    raise SystemExit(main())
