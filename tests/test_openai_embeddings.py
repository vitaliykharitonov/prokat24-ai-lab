from types import SimpleNamespace
from unittest.mock import Mock

from prokat24_ai_lab.openai_embeddings import create_openai_embed_text


def test_create_openai_embed_text_returns_embedding() -> None:
    client = Mock()

    client.embeddings.create.return_value = SimpleNamespace(
        data=[
            SimpleNamespace(
                embedding=[0.1, -0.2, 0.3],
            )
        ]
    )

    embed_text = create_openai_embed_text(client)

    vector = embed_text("доставка по києву")

    assert vector == [0.1, -0.2, 0.3]

    client.embeddings.create.assert_called_once_with(
        model="text-embedding-3-small",
        input="доставка по києву",
    )
