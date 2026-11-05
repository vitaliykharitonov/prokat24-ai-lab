from openai import OpenAI

from prokat24_ai_lab.embeddings.openai_client import (
    OPENAI_EMBEDDING_MODEL,
    create_openai_embed_text,
)
from prokat24_ai_lab.embeddings.query_index import (
    build_query_embedding_index,
    save_query_embedding_index,
)
from prokat24_ai_lab.evals.retrieval.cases import load_retrieval_eval_cases
from prokat24_ai_lab.project_paths import (
    QUERY_EMBEDDING_INDEX_FILE,
    RETRIEVAL_EVAL_CASES_FILE,
)


def main() -> None:
    eval_cases = load_retrieval_eval_cases(RETRIEVAL_EVAL_CASES_FILE)
    queries: list[str] = []

    for case in eval_cases:
        queries.append(case["query"])

    client = OpenAI()

    embed_text = create_openai_embed_text(client)

    query_embedding_index = build_query_embedding_index(queries, embed_text, OPENAI_EMBEDDING_MODEL)

    save_query_embedding_index(query_embedding_index, QUERY_EMBEDDING_INDEX_FILE)

    print(f"Model: {query_embedding_index['model']}")
    print(f"Dimensions: {query_embedding_index['dimensions']}")
    print(
        f"Saved {len(query_embedding_index['embeddings'])} "
        f"query embeddings to {QUERY_EMBEDDING_INDEX_FILE}"
    )


if __name__ == "__main__":
    main()
