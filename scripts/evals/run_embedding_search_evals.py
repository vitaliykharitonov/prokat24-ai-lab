import datetime

from openai import OpenAI

from prokat24_ai_lab.embeddings.document_index import load_document_embedding_index
from prokat24_ai_lab.embeddings.openai_client import create_openai_embed_text
from prokat24_ai_lab.evals.embedding_evals import (
    build_embedding_search_eval_report,
    load_embedding_search_eval_cases,
    run_embedding_search_evals,
    save_embedding_search_eval_report,
)
from prokat24_ai_lab.git_metadata import get_current_git_commit, is_git_worktree_dirty
from prokat24_ai_lab.project_paths import (
    DOCUMENT_EMBEDDING_INDEX_FILE,
    EMBEDDING_SEARCH_EVAL_CASES_FILE,
    EMBEDDING_SEARCH_EVAL_REPORT_FILE,
)


def main() -> None:
    document_embedding_index = load_document_embedding_index(DOCUMENT_EMBEDDING_INDEX_FILE)
    eval_cases = load_embedding_search_eval_cases(EMBEDDING_SEARCH_EVAL_CASES_FILE)
    client = OpenAI()

    embed_text = create_openai_embed_text(client, document_embedding_index["model"])

    results = run_embedding_search_evals(
        eval_cases,
        document_embedding_index["embeddings"],
        embed_text,
    )

    created_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    git_commit = get_current_git_commit()
    git_dirty = is_git_worktree_dirty()

    report = build_embedding_search_eval_report(
        results,
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    save_embedding_search_eval_report(report, EMBEDDING_SEARCH_EVAL_REPORT_FILE)

    for result in results:
        print(f"Query: {result['query']}")
        print(f"Expected document: {result['expected_document_id']}")
        print(f"Actual document: {result['actual_document_id']}")
        print(f"Actual score: {result['actual_score']}")
        print(f"Passed: {result['passed']}")

        if result["passed"] is False:
            print("Top results:")

            for index, (document_id, score) in enumerate(result["top_results"], start=1):
                print(f"{index}. {document_id} - {score:.4f}")

        print()

    print(f"Passed: {report['passed_cases']} / {report['total_cases']}")
    print(f"Accuracy: {report['accuracy']:.2%}")
    print(f"Top-3 accuracy: {report['top_3_accuracy']:.2%}")
    print(f"Report saved to {EMBEDDING_SEARCH_EVAL_REPORT_FILE}")


if __name__ == "__main__":
    main()
