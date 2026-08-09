from openai import OpenAI

from prokat24_ai_lab.evals.semantic_grading import JudgeInput, Rubric
from prokat24_ai_lab.judges.openai_judge import create_openai_judge


def main() -> None:
    client = OpenAI()
    judge = create_openai_judge(client)
    judge_input: JudgeInput = {
        "question": "Чи можна залишити паспорт як заставу?",
        "context": "Громадянський паспорт не можна залишати як заставу.",
        "answer": "Так, паспорт можна залишити як заставу.",
    }
    rubric: Rubric = {
        "name": "delivery_answer",
        "criteria": [
            {
                "name": "correctness",
                "description": "The answer must correctly state the delivery price.",
            },
            {
                "name": "groundedness",
                "description": "The answer must use only information from the provided context.",
            },
        ],
    }

    result = judge(judge_input, rubric)

    print(result)


if __name__ == "__main__":
    main()
