from prokat24_ai_lab.embeddings.document_index import load_document_embedding_index
from prokat24_ai_lab.embeddings.query_index import load_query_embedding_index
from prokat24_ai_lab.evals.retrieval.cases import load_retrieval_eval_cases
from prokat24_ai_lab.evals.retrieval.hybrid import (
    HybridSearchEvalResult,
    build_hybrid_search_eval_summary,
    run_weighted_hybrid_search_evals,
)
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.project_paths import (
    DOCUMENT_EMBEDDING_INDEX_FILE,
    KNOWLEDGE_PAGES_FILE,
    QUERY_EMBEDDING_INDEX_FILE,
    RETRIEVAL_EVAL_CASES_FILE,
)

RANK_CONSTANT = 60
WEIGHT_CONFIGURATIONS = [
    (1.0, 1.0),
    (1.0, 1.25),
    (1.0, 1.5),
    (1.0, 2.0),
    (1.25, 1.0),
    (1.5, 1.0),
    (2.0, 1.0),
    (1.0, 0.0),
    (0.0, 1.0),
]


def main() -> int:
    knowledge_pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    cases = load_retrieval_eval_cases(RETRIEVAL_EVAL_CASES_FILE)
    document_embedding_index = load_document_embedding_index(DOCUMENT_EMBEDDING_INDEX_FILE)
    query_embedding_index = load_query_embedding_index(QUERY_EMBEDDING_INDEX_FILE)

    if document_embedding_index["model"] != query_embedding_index["model"]:
        raise ValueError("Embedding models do not match")

    if document_embedding_index["dimensions"] != query_embedding_index["dimensions"]:
        raise ValueError("Embedding dimensions do not match")

    print(f"{'Keyword weight':>14} | {'Embedding weight':>16} | {'Top-1':>17} | {'Top-3':>17}")
    print("-" * 73)

    baseline_results: list[HybridSearchEvalResult] | None = None

    weighted_results_by_configuration: list[tuple[float, float, list[HybridSearchEvalResult]]] = []

    for keyword_weight, embedding_weight in WEIGHT_CONFIGURATIONS:
        results = run_weighted_hybrid_search_evals(
            cases,
            knowledge_pages,
            document_embedding_index["embeddings"],
            query_embedding_index["embeddings"],
            keyword_weight,
            embedding_weight,
            rank_constant=RANK_CONSTANT,
        )

        summary = build_hybrid_search_eval_summary(results)

        top_1 = f"{summary['passed_cases']}/{summary['total_cases']} {summary['accuracy']:.2%}"
        top_3 = (
            f"{summary['top_3_passed_cases']}/{summary['total_cases']} "
            f"{summary['top_3_accuracy']:.2%}"
        )

        print(f"{keyword_weight:>14.2f} | {embedding_weight:>16.2f} | {top_1:>17} | {top_3:>17}")

        if baseline_results is None:
            baseline_results = results

        else:
            weighted_results_by_configuration.append(
                (
                    keyword_weight,
                    embedding_weight,
                    results,
                )
            )

    if baseline_results is None:
        raise ValueError("Weight configurations cannot be empty")

    print()
    print("CHANGES COMPARED WITH WEIGHTS 1.00:1.00")

    for keyword_weight, embedding_weight, weighted_results in weighted_results_by_configuration:
        print()
        print("-" * 80)
        print(f"Weights: {keyword_weight:.2f}:{embedding_weight:.2f}")
        print("-" * 80)

        changed_cases = 0

        for baseline_result, weighted_result in zip(
            baseline_results,
            weighted_results,
            strict=True,
        ):
            if baseline_result["actual_document_id"] == weighted_result["actual_document_id"]:
                continue

            changed_cases += 1

            print(f"Query: {weighted_result['query']}")
            print(f"Expected: {weighted_result['expected_document_id']}")
            print(
                "1.00:1.00 actual: "
                f"{baseline_result['actual_document_id']} "
                f"| passed: {baseline_result['passed']}"
            )
            print(
                f"{keyword_weight:.2f}:{embedding_weight:.2f} actual: "
                f"{weighted_result['actual_document_id']} "
                f"| passed: {weighted_result['passed']}"
            )
            print()

        if changed_cases == 0:
            print("No Top-1 changes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
