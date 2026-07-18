from collections.abc import Callable

from openai import OpenAI


def create_openai_embed_text(
    client: OpenAI,
    model: str = "text-embedding-3-small",
) -> Callable[[str], list[float]]:
    def embed_text(text: str) -> list[float]:
        response = client.embeddings.create(model=model, input=text)

        return response.data[0].embedding

    return embed_text
