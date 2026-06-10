from argparse import ArgumentParser, Namespace

from prokat24_ai_lab.answer_engine import answer_question
from prokat24_ai_lab.knowledge_base import load_knowledge_base
from prokat24_ai_lab.project_paths import KNOWLEDGE_PAGES_FILE


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Answer to the question")

    parser.add_argument(
        "question",
        help="Question",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    answer = answer_question(args.question, pages)

    print(f"Question: {args.question}")
    print(f"Confidence: {answer['confidence']}")
    print()
    print("Answer:")
    print(f"{answer['answer']}")
    print()
    print("Sources:")

    for index, source in enumerate(answer["sources"], start=1):
        print(f"{index}. {source['title']}")
        print(f"URL: {source['url']}")


if __name__ == "__main__":
    main()
