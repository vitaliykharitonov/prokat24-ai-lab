from prokat24_ai_lab.embeddings.document_index import load_document_embedding_index
from prokat24_ai_lab.embeddings.query_index import load_query_embedding_index
from prokat24_ai_lab.evals.embedding_evals import load_embedding_search_eval_cases
from prokat24_ai_lab.evals.retrieval_comparison import (
    build_retrieval_comparison_summary,
    run_retrieval_comparison_evals,
)
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.project_paths import (
    DOCUMENT_EMBEDDING_INDEX_FILE,
    EMBEDDING_SEARCH_EVAL_CASES_FILE,
    KNOWLEDGE_PAGES_FILE,
    QUERY_EMBEDDING_INDEX_FILE,
)


def main() -> int:
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    cases = load_embedding_search_eval_cases(EMBEDDING_SEARCH_EVAL_CASES_FILE)
    document_embedding_index = load_document_embedding_index(DOCUMENT_EMBEDDING_INDEX_FILE)
    query_embedding_index = load_query_embedding_index(QUERY_EMBEDDING_INDEX_FILE)

    if document_embedding_index["model"] != query_embedding_index["model"]:
        raise ValueError("Embedding models do not match")

    if document_embedding_index["dimensions"] != query_embedding_index["dimensions"]:
        raise ValueError("Embedding dimensions do not match")

    results = run_retrieval_comparison_evals(
        cases,
        pages,
        document_embedding_index["embeddings"],
        query_embedding_index["embeddings"],
    )
    summary = build_retrieval_comparison_summary(results)

    for result in results:
        if result["keyword_passed"] and result["embedding_passed"]:
            continue

        print(f"Query: {result['query']}")
        print(f"Expected document: {result['expected_document_id']}")
        print()
        print(f"Keyword document: {result['keyword_actual_document_id']}")
        print(f"Keyword expected rank: {result['keyword_expected_rank']}")
        print(f"Keyword passed: {result['keyword_passed']}")
        print()
        print(f"Embedding document: {result['embedding_actual_document_id']}")
        print(f"Embedding expected rank: {result['embedding_expected_rank']}")
        print(f"Embedding passed: {result['embedding_passed']}")

    print()
    print()
    print(f"Total cases: {summary['total_cases']}")
    print()
    print(f"Keyword passed cases: {summary['keyword_passed_cases']}")
    print(f"Keyword accuracy: {summary['keyword_accuracy']:.2%}")
    print(f"Keyword top-3 accuracy: {summary['keyword_top_3_accuracy']:.2%}")
    print()
    print(f"Embedding passed cases: {summary['embedding_passed_cases']}")
    print(f"Embedding accuracy: {summary['embedding_accuracy']:.2%}")
    print(f"Embedding top-3 accuracy: {summary['embedding_top_3_accuracy']:.2%}")
    print()
    print(f"Both passed cases: {summary['both_passed_cases']}")
    print(f"Keyword only passed cases: {summary['keyword_only_passed_cases']}")
    print(f"Embedding only passed cases: {summary['embedding_only_passed_cases']}")
    print(f"Both failed cases: {summary['both_failed_cases']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
