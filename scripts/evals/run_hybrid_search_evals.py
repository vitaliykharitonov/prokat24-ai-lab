from prokat24_ai_lab.embeddings.document_index import load_document_embedding_index
from prokat24_ai_lab.embeddings.query_index import load_query_embedding_index
from prokat24_ai_lab.evals.embedding_evals import load_embedding_search_eval_cases
from prokat24_ai_lab.evals.hybrid_search_evals import (
    build_hybrid_search_eval_summary,
    run_hybrid_search_evals,
)
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.project_paths import (
    DOCUMENT_EMBEDDING_INDEX_FILE,
    EMBEDDING_SEARCH_EVAL_CASES_FILE,
    KNOWLEDGE_PAGES_FILE,
    QUERY_EMBEDDING_INDEX_FILE,
)

RANK_CONSTANT = 60


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

    summary = build_hybrid_search_eval_summary(results)

    for result in results:
        print(f"Query: {result['query']}")
        print(f"Expected document: {result['expected_document_id']}")
        print(f"Actual document: {result['actual_document_id']}")
        print(f"Actual score: {result['actual_score']}")
        print(f"Expected rank: {result['expected_rank']}")
        print(f"Passed: {result['passed']}")
        print()

    print(f"Total cases: {summary['total_cases']}")
    print(f"Passed cases: {summary['passed_cases']}")
    print(f"Accuracy: {summary['accuracy']:.2%}")
    print(f"Top 3 passed cases: {summary['top_3_passed_cases']}")
    print(f"Top 3 accuracy: {summary['top_3_accuracy']:.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
