from unittest.mock import Mock

from openai import OpenAI

from prokat24_ai_lab.rag.openai_client import create_openai_generate_text


def test_create_openai_generate_text_calls_responses_api() -> None:
    instructions = "Use only the provided context."
    prompt_input = "Context and question"
    expected_answer = "Спорядження можна отримати доставкою."

    client = Mock(spec=OpenAI)

    client.responses.create.return_value.output_text = expected_answer

    generate_text = create_openai_generate_text(
        client=client,
        model="test-model",
    )

    result = generate_text(instructions, prompt_input)

    assert result == expected_answer

    client.responses.create.assert_called_once_with(
        model="test-model",
        instructions=instructions,
        input=prompt_input,
    )
