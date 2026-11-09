import json
from pathlib import Path
from typing import Literal, TypedDict

from prokat24_ai_lab.evals.grading.comparison import GraderComparisonResult
from prokat24_ai_lab.evals.report_metadata import build_eval_report_metadata


class GraderComparisonReport(TypedDict):
    report_type: str
    schema_version: int
    created_at: str
    git_commit: str | None
    git_dirty: bool | None
    run_id: str
    split: Literal["dev", "holdout"]
    run_count: int
    runs: list[list[GraderComparisonResult]]


def build_grader_comparison_report(
    all_runs: list[list[GraderComparisonResult]],
    split: Literal["dev", "holdout"],
    run_id: str,
    created_at: str,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> GraderComparisonReport:
    metadata = build_eval_report_metadata(
        report_type="grader_comparison_report",
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    return GraderComparisonReport(
        **metadata,
        run_id=run_id,
        split=split,
        run_count=len(all_runs),
        runs=all_runs,
    )


def save_grader_comparison_report(
    report: GraderComparisonReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    split_path = output_dir / report["split"]
    historical_path = split_path / "runs"

    historical_file = historical_path / f"{report['run_id']}.json"
    latest_file = split_path / "latest.json"

    historical_path.mkdir(parents=True, exist_ok=True)

    with open(historical_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    with open(latest_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    return historical_file, latest_file
