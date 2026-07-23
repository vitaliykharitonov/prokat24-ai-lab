from typing import TypedDict


class EvalReportMetadata(TypedDict):
    report_type: str
    schema_version: int
    created_at: str | None
    git_commit: str | None
    git_dirty: bool | None


def build_eval_report_metadata(
    report_type: str,
    created_at: str,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> EvalReportMetadata:
    return {
        "report_type": report_type,
        "schema_version": 1,
        "created_at": created_at,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
