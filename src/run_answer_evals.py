import datetime

from prokat24_ai_lab.answer_evals import (
    build_answer_eval_report,
    load_answer_eval_cases,
    run_answer_evals,
    save_answer_eval_report,
)
from prokat24_ai_lab.eval_exit_codes import get_eval_exit_code
from prokat24_ai_lab.git_metadata import get_current_git_commit, is_git_worktree_dirty
from prokat24_ai_lab.knowledge_base import load_knowledge_base
from prokat24_ai_lab.project_paths import (
    ANSWER_EVAL_CASES_FILE,
    ANSWER_EVAL_REPORT_FILE,
    KNOWLEDGE_PAGES_FILE,
)


def main() -> int:
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    eval_cases = load_answer_eval_cases(ANSWER_EVAL_CASES_FILE)

    answer_results = run_answer_evals(eval_cases, pages)

    created_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    git_commit = get_current_git_commit()
    git_dirty = is_git_worktree_dirty()

    report = build_answer_eval_report(
        answer_results,
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    print(f"Answer evals: {report['passed_cases']}/{report['total_cases']} passed")

    save_answer_eval_report(report, ANSWER_EVAL_REPORT_FILE)

    print(f"Saved report to {ANSWER_EVAL_REPORT_FILE}")

    failed_cases = report["total_cases"] - report["passed_cases"]

    return get_eval_exit_code(failed_cases)


if __name__ == "__main__":
    raise SystemExit(main())
