import prokat24_ai_lab.evals.report_metadata as eval_report_metadata


def test_build_eval_report_metadata_returns_common_report_fields() -> None:
    metadata = eval_report_metadata.build_eval_report_metadata(
        report_type="search_eval_report",
        created_at="2026-01-01T00:00:00+00:00",
        git_commit="abc1234",
        git_dirty=True,
    )

    assert metadata == {
        "report_type": "search_eval_report",
        "schema_version": 1,
        "created_at": "2026-01-01T00:00:00+00:00",
        "git_commit": "abc1234",
        "git_dirty": True,
    }
