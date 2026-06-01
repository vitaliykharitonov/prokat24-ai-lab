from prokat24_ai_lab.knowledge_base import build_knowledge_base, save_knowledge_base
from prokat24_ai_lab.project_paths import KNOWLEDGE_PAGES_FILE


def main() -> None:
    knowledge_base = build_knowledge_base()
    save_knowledge_base(knowledge_base, KNOWLEDGE_PAGES_FILE)

    print(f"Exported {len(knowledge_base)} knowledge pages to {KNOWLEDGE_PAGES_FILE}")


if __name__ == "__main__":
    main()