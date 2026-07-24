from argparse import ArgumentParser, Namespace

from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.knowledge.keyword_search import search_knowledge_base
from prokat24_ai_lab.project_paths import KNOWLEDGE_PAGES_FILE


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Search Prokat24 knowledge base")

    parser.add_argument(
        "query",
        help="Search query",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of results to show",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    results = search_knowledge_base(args.query, pages)

    print(f"Query: {args.query}")
    print()

    if not results:
        print("No results found")
        return

    for index, result in enumerate(results[: args.limit], start=1):
        print(f"{index}. {result['title']}")
        print(f"Score: {result['score']}")
        print(f"Matched words: {result['matched_words']}")
        print(f"URL: {result['url']}")
        print("Snippet:")
        print(result["snippet"])
        print()


if __name__ == "__main__":
    main()
