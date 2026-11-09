import json
from pathlib import Path

from prokat24_ai_lab.evals.grading.report import (
    build_grader_comparison_report,
    save_grader_comparison_report,
)
from tests.evals.grading.fixtures import GRADER_COMPARISON_TEST_RESULTS


def test_build_grader_comparison_report() -> None:
    all_runs = [
        GRADER_COMPARISON_TEST_RESULTS,
        GRADER_COMPARISON_TEST_RESULTS,
    ]

    report = build_grader_comparison_report(
        all_runs=all_runs,
        split="holdout",
        run_id="test_run",
        created_at="2026-10-04T16:00:00Z",
        git_commit="abc123",
        git_dirty=False,
    )

    assert report["report_type"] == "grader_comparison_report"
    assert report["schema_version"] == 1
    assert report["created_at"] == "2026-10-04T16:00:00Z"
    assert report["git_commit"] == "abc123"
    assert report["git_dirty"] is False
    assert report["run_id"] == "test_run"
    assert report["split"] == "holdout"
    assert report["run_count"] == 2
    assert report["runs"] == all_runs


def test_save_grader_comparison_report_saves_historical_and_latest_files(
    tmp_path: Path,
) -> None:
    report = build_grader_comparison_report(
        all_runs=[GRADER_COMPARISON_TEST_RESULTS],
        split="holdout",
        run_id="test_run",
        created_at="2026-10-04T16:00:00Z",
        git_commit="abc123",
        git_dirty=False,
    )

    historical_file, latest_file = save_grader_comparison_report(
        report=report,
        output_dir=tmp_path,
    )

    expected_historical_file = tmp_path / "holdout" / "runs" / "test_run.json"
    expected_latest_file = tmp_path / "holdout" / "latest.json"

    assert historical_file == expected_historical_file
    assert latest_file == expected_latest_file

    assert historical_file.exists()
    assert latest_file.exists()

    with open(historical_file, encoding="utf-8") as file:
        historical_report = json.load(file)

    with open(latest_file, encoding="utf-8") as file:
        latest_report = json.load(file)

    assert historical_report == report
    assert latest_report == report
