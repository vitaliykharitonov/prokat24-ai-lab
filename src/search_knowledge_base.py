from prokat24_ai_lab.knowledge_base import load_knowledge_base
from prokat24_ai_lab.knowledge_search import search_knowledge_base
from prokat24_ai_lab.project_paths import KNOWLEDGE_PAGES_FILE


def main() -> None:
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)

    query = "доставка київ"

    results = search_knowledge_base(query, pages)

    print(f"Query: {query}")
    print()

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['title']}")
        print(f"Score: {result['score']}")
        print(f"URL: {result['url']}")
        print("Snippet:")
        print(result["snippet"])
        print()


if __name__ == "__main__":
    main()