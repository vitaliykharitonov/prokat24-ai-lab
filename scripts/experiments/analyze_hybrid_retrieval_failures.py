from prokat24_ai_lab.embeddings.document_index import load_document_embedding_index
from prokat24_ai_lab.embeddings.query_index import load_query_embedding_index
from prokat24_ai_lab.embeddings.search import rank_query_embedding_by_document_embeddings
from prokat24_ai_lab.evals.embedding_evals import (
    load_embedding_search_eval_cases,
)
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.knowledge.keyword_search import (
    KnowledgeSearchResult,
    get_query_words,
    search_knowledge_base,
)
from prokat24_ai_lab.project_paths import (
    DOCUMENT_EMBEDDING_INDEX_FILE,
    EMBEDDING_SEARCH_EVAL_CASES_FILE,
    KNOWLEDGE_PAGES_FILE,
    QUERY_EMBEDDING_INDEX_FILE,
)
from prokat24_ai_lab.retrieval.hybrid_search import rank_query_with_reciprocal_rank_fusion

RANK_CONSTANT = 60


def find_document_rank(document_ids: list[str], expected_document_id: str) -> int | None:
    for index, document_id in enumerate(document_ids, start=1):
        if document_id == expected_document_id:
            return index

    return None


def print_top_documents(title: str, document_ids: list[str], expected_document_id: str) -> None:
    print(f"{title} Top-3:")

    for index, document_id in enumerate(document_ids[:3], start=1):
        print(f"{index}. {document_id}")

    expected_rank = find_document_rank(
        document_ids,
        expected_document_id,
    )

    print(f"Expected rank: {expected_rank}")


def print_top_scored_documents(
    title: str,
    results: list[tuple[str, float]],
    expected_document_id: str,
) -> None:
    print(f"{title} Top-4:")

    for index, (document_id, score) in enumerate(results[:4], start=1):
        print(f"{index}. {document_id} — {score:.6f}")

    document_ids = [document_id for document_id, _score in results]

    expected_rank = find_document_rank(document_ids, expected_document_id)

    print(f"Expected rank: {expected_rank}")


def print_top_keyword_documents(
    results: list[KnowledgeSearchResult],
    expected_document_id: str,
) -> None:
    for index, result in enumerate(results[:3], start=1):
        print(f"{index}. {result['url']} — score: {result['score']:.2f}")
        print(f"   Matched words: {result['matched_words']}")

    for word, detail in result["score_details"].items():
        print(
            f"   {word}: "
            f"count={detail['count']}, "
            f"weight={detail['weight']:.2f}, "
            f"contribution={detail['contribution']:.2f}"
        )

    document_ids = [result["url"] for result in results]

    expected_rank = find_document_rank(
        document_ids,
        expected_document_id,
    )

    print(f"Expected rank: {expected_rank}")


def main() -> int:
    knowledge_pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    cases = load_embedding_search_eval_cases(EMBEDDING_SEARCH_EVAL_CASES_FILE)
    document_embedding_index = load_document_embedding_index(DOCUMENT_EMBEDDING_INDEX_FILE)
    query_embedding_index = load_query_embedding_index(QUERY_EMBEDDING_INDEX_FILE)

    if document_embedding_index["model"] != query_embedding_index["model"]:
        raise ValueError("Embedding models do not match")

    if document_embedding_index["dimensions"] != query_embedding_index["dimensions"]:
        raise ValueError("Embedding dimensions do not match")

    for case in cases:
        query = case["query"]
        expected_document_id = case["expected_document_id"]
        query_embedding = query_embedding_index["embeddings"][query]

        keyword_results = search_knowledge_base(query, knowledge_pages)
        embedding_results = rank_query_embedding_by_document_embeddings(
            query_embedding,
            document_embedding_index["embeddings"],
        )
        hybrid_results = rank_query_with_reciprocal_rank_fusion(
            query,
            knowledge_pages,
            document_embedding_index["embeddings"],
            query_embedding,
            rank_constant=RANK_CONSTANT,
        )

        actual_document_id = hybrid_results[0][0] if hybrid_results else None

        if actual_document_id == expected_document_id:
            continue

        print("-" * 80)
        print(f"Query: {query}")
        print(f"Query words: {get_query_words(query)}")
        print(f"Expected: {expected_document_id}")
        print()

        print_top_keyword_documents(
            keyword_results,
            expected_document_id,
        )

        print()
        print_top_scored_documents(
            "Embedding",
            embedding_results,
            expected_document_id,
        )

        print()
        print_top_scored_documents(
            "Hybrid",
            hybrid_results,
            expected_document_id,
        )

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
