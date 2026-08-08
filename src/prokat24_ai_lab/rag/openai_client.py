from openai import OpenAI

from prokat24_ai_lab.rag.generation import TextGenerator

OPENAI_TEXT_MODEL = "gpt-5-mini"


def create_openai_generate_text(
    client: OpenAI,
    model: str = OPENAI_TEXT_MODEL,
) -> TextGenerator:
    def generate_text(instructions: str, prompt_input: str) -> str:
        response = client.responses.create(
            model=model,
            instructions=instructions,
            input=prompt_input,
        )

        return response.output_text

    return generate_text
