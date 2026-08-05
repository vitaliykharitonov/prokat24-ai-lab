from argparse import ArgumentParser, Namespace

from openai import OpenAI

from prokat24_ai_lab.embeddings.document_index import (
    load_document_embedding_index,
)
from prokat24_ai_lab.embeddings.openai_client import create_openai_embed_text
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.project_paths import DOCUMENT_EMBEDDING_INDEX_FILE, KNOWLEDGE_PAGES_FILE
from prokat24_ai_lab.rag.answer import answer_question
from prokat24_ai_lab.rag.openai_client import create_openai_generate_text
from prokat24_ai_lab.rag.ranker import create_hybrid_document_ranker


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Answer to the question")
    parser.add_argument("query", help="Search query")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Maximum number of context documents",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)

    document_embedding_index = load_document_embedding_index(DOCUMENT_EMBEDDING_INDEX_FILE)

    client = OpenAI()

    model = document_embedding_index["model"]

    embed_text = create_openai_embed_text(client=client, model=model)

    rank_documents = create_hybrid_document_ranker(
        pages=pages,
        document_embeddings=document_embedding_index["embeddings"],
        embed_text=embed_text,
    )

    generate_text = create_openai_generate_text(client)

    answer = answer_question(
        question=args.query,
        pages=pages,
        rank_documents=rank_documents,
        generate_text=generate_text,
        top_k=args.top_k,
    )

    print(answer)


if __name__ == "__main__":
    main()
