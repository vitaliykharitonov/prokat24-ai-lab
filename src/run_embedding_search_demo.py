from openai import OpenAI

from prokat24_ai_lab.embedding_search import rank_texts_by_embeddings
from prokat24_ai_lab.openai_embeddings import create_openai_embed_text


def main() -> None:
    client = OpenAI()

    embed_text = create_openai_embed_text(client)

    query = "привезти спорядження додому"

    documents = {
        "delivery": "Доставка спорядження по Києву",
        "contacts": "Адреса та контакти пункту прокату",
        "deposit": "Умови внесення застави за спорядження",
    }

    results = rank_texts_by_embeddings(query, documents, embed_text)

    for document_id, score in results:
        print(f"{document_id}: {score:.4f}")


if __name__ == "__main__":
    main()
