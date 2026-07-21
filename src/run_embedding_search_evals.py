from openai import OpenAI

from prokat24_ai_lab.embedding_evals import (
    load_embedding_search_eval_cases,
    run_embedding_search_evals,
)
from prokat24_ai_lab.embedding_index import load_document_embedding_index
from prokat24_ai_lab.openai_embeddings import create_openai_embed_text
from prokat24_ai_lab.project_paths import (
    DOCUMENT_EMBEDDING_INDEX_FILE,
    EMBEDDING_SEARCH_EVAL_CASES_FILE,
)


def main() -> None:
    document_embeddings = load_document_embedding_index(DOCUMENT_EMBEDDING_INDEX_FILE)
    eval_cases = load_embedding_search_eval_cases(EMBEDDING_SEARCH_EVAL_CASES_FILE)
    client = OpenAI()

    embed_text = create_openai_embed_text(client)

    results = run_embedding_search_evals(eval_cases, document_embeddings, embed_text)

    for result in results:
        print(f"Query: {result['query']}")
        print(f"Expected document: {result['expected_document_id']}")
        print(f"Actual document: {result['actual_document_id']}")
        print(f"Actual score: {result['actual_score']}")
        print(f"Passed: {result['passed']}")

        if result["passed"] is False:
            print("Top results:")

            for index, (document_id, score) in enumerate(result["top_results"], start=1):
                print(f"{index}. {document_id} - {score:4f}")

        print()


if __name__ == "__main__":
    main()
