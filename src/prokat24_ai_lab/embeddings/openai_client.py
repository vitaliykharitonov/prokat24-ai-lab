from collections.abc import Callable

from openai import OpenAI

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"


def create_openai_embed_text(
    client: OpenAI,
    model: str = OPENAI_EMBEDDING_MODEL,
) -> Callable[[str], list[float]]:
    def embed_text(text: str) -> list[float]:
        response = client.embeddings.create(model=model, input=text)

        return response.data[0].embedding

    return embed_text
