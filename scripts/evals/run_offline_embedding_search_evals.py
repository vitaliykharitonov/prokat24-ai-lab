import datetime

from prokat24_ai_lab.embeddings.document_index import load_document_embedding_index
from prokat24_ai_lab.embeddings.query_index import load_query_embedding_index
from prokat24_ai_lab.evals.embedding_evals import (
    build_embedding_search_eval_report,
    load_embedding_search_eval_cases,
    run_precomputed_embedding_search_evals,
    save_embedding_search_eval_report,
)
from prokat24_ai_lab.git_metadata import get_current_git_commit, is_git_worktree_dirty
from prokat24_ai_lab.project_paths import (
    DOCUMENT_EMBEDDING_INDEX_FILE,
    EMBEDDING_SEARCH_EVAL_CASES_FILE,
    OFFLINE_EMBEDDING_SEARCH_EVAL_REPORT_FILE,
    QUERY_EMBEDDING_INDEX_FILE,
)

MIN_ACCURACY = 0.80
MIN_TOP_3_ACCURACY = 1.00


def main() -> int:
    document_embedding_index = load_document_embedding_index(DOCUMENT_EMBEDDING_INDEX_FILE)
    query_embedding_index = load_query_embedding_index(QUERY_EMBEDDING_INDEX_FILE)
    eval_cases = load_embedding_search_eval_cases(EMBEDDING_SEARCH_EVAL_CASES_FILE)

    if document_embedding_index["model"] != query_embedding_index["model"]:
        raise ValueError("Embedding models do not match")

    if document_embedding_index["dimensions"] != query_embedding_index["dimensions"]:
        raise ValueError("Embedding dimensions do not match")

    results = run_precomputed_embedding_search_evals(
        eval_cases,
        document_embedding_index["embeddings"],
        query_embedding_index["embeddings"],
    )

    created_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    git_commit = get_current_git_commit()
    git_dirty = is_git_worktree_dirty()

    report = build_embedding_search_eval_report(
        results,
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    save_embedding_search_eval_report(report, OFFLINE_EMBEDDING_SEARCH_EVAL_REPORT_FILE)

    for result in results:
        print(f"Query: {result['query']}")
        print(f"Expected document: {result['expected_document_id']}")
        print(f"Actual document: {result['actual_document_id']}")
        print(f"Actual score: {result['actual_score']}")
        print(f"Passed: {result['passed']}")

        if result["passed"] is False:
            print("Top results:")

            for index, (document_id, score) in enumerate(result["top_results"], start=1):
                print(f"{index}. {document_id} - {score:.4f}")

        print()

    print(f"Passed: {report['passed_cases']} / {report['total_cases']}")
    print(f"Accuracy: {report['accuracy']:.2%}")
    print(f"Top-3 accuracy: {report['top_3_accuracy']:.2%}")
    print(f"Report saved to {OFFLINE_EMBEDDING_SEARCH_EVAL_REPORT_FILE}")

    if report["accuracy"] < MIN_ACCURACY:
        return 1

    if report["top_3_accuracy"] < MIN_TOP_3_ACCURACY:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
