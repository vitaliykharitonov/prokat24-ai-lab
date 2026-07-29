import datetime

from prokat24_ai_lab.embeddings.document_index import load_document_embedding_index
from prokat24_ai_lab.embeddings.query_index import load_query_embedding_index
from prokat24_ai_lab.evals.embedding_evals import load_embedding_search_eval_cases
from prokat24_ai_lab.evals.hybrid_search_evals import (
    build_hybrid_search_eval_report,
    run_hybrid_search_evals,
    save_hybrid_search_eval_report,
)
from prokat24_ai_lab.git_metadata import get_current_git_commit, is_git_worktree_dirty
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.project_paths import (
    DOCUMENT_EMBEDDING_INDEX_FILE,
    EMBEDDING_SEARCH_EVAL_CASES_FILE,
    HYBRID_SEARCH_EVAL_REPORT_FILE,
    KNOWLEDGE_PAGES_FILE,
    QUERY_EMBEDDING_INDEX_FILE,
)

RANK_CONSTANT = 60
MIN_ACCURACY = 0.80
MIN_TOP_3_ACCURACY = 0.90


def main() -> int:
    knowledge_pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    cases = load_embedding_search_eval_cases(EMBEDDING_SEARCH_EVAL_CASES_FILE)
    document_embedding_index = load_document_embedding_index(DOCUMENT_EMBEDDING_INDEX_FILE)
    query_embedding_index = load_query_embedding_index(QUERY_EMBEDDING_INDEX_FILE)

    if document_embedding_index["model"] != query_embedding_index["model"]:
        raise ValueError("Embedding models do not match")

    if document_embedding_index["dimensions"] != query_embedding_index["dimensions"]:
        raise ValueError("Embedding dimensions do not match")

    results = run_hybrid_search_evals(
        cases,
        knowledge_pages,
        document_embedding_index["embeddings"],
        query_embedding_index["embeddings"],
        rank_constant=RANK_CONSTANT,
    )

    created_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    git_commit = get_current_git_commit()
    git_dirty = is_git_worktree_dirty()

    report = build_hybrid_search_eval_report(
        results,
        created_at=created_at,
        rank_constant=RANK_CONSTANT,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    save_hybrid_search_eval_report(report, HYBRID_SEARCH_EVAL_REPORT_FILE)

    for result in results:
        print(f"Query: {result['query']}")
        print(f"Expected document: {result['expected_document_id']}")
        print(f"Actual document: {result['actual_document_id']}")
        print(f"Actual score: {result['actual_score']}")
        print(f"Expected rank: {result['expected_rank']}")
        print(f"Passed: {result['passed']}")
        print()

    print(f"Rank constant: {report['rank_constant']}")
    print(f"Total cases: {report['total_cases']}")
    print(f"Passed cases: {report['passed_cases']}")
    print(f"Accuracy: {report['accuracy']:.2%}")
    print(f"Top 3 passed cases: {report['top_3_passed_cases']}")
    print(f"Top 3 accuracy: {report['top_3_accuracy']:.2%}")
    print(f"Report saved to {HYBRID_SEARCH_EVAL_REPORT_FILE}")

    if report["accuracy"] < MIN_ACCURACY:
        return 1

    if report["top_3_accuracy"] < MIN_TOP_3_ACCURACY:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
