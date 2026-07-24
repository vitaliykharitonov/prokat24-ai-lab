from openai import OpenAI

from prokat24_ai_lab.embeddings.document_index import (
    build_document_embedding_index,
    save_document_embedding_index,
)
from prokat24_ai_lab.embeddings.openai_client import (
    OPENAI_EMBEDDING_MODEL,
    create_openai_embed_text,
)
from prokat24_ai_lab.embeddings.search import build_embedding_documents
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.project_paths import DOCUMENT_EMBEDDING_INDEX_FILE, KNOWLEDGE_PAGES_FILE


def main() -> None:
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    documents = build_embedding_documents(pages)
    client = OpenAI()
    embed_text = create_openai_embed_text(client)

    document_embedding_index = build_document_embedding_index(
        documents,
        embed_text,
        OPENAI_EMBEDDING_MODEL,
    )

    save_document_embedding_index(document_embedding_index, DOCUMENT_EMBEDDING_INDEX_FILE)

    print(
        f"Saved {len(document_embedding_index['embeddings'])} "
        f"document embeddings to {DOCUMENT_EMBEDDING_INDEX_FILE}"
    )


if __name__ == "__main__":
    main()
