import datetime

from openai import OpenAI

from prokat24_ai_lab.embeddings.document_index import load_document_embedding_index
from prokat24_ai_lab.embeddings.openai_client import create_openai_embed_text
from prokat24_ai_lab.evals.rag_answer_evals import (
    build_rag_answer_eval_report,
    load_rag_answer_eval_cases,
    run_rag_answer_evals,
    save_rag_answer_eval_report,
)
from prokat24_ai_lab.git_metadata import get_current_git_commit, is_git_worktree_dirty
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.project_paths import (
    DOCUMENT_EMBEDDING_INDEX_FILE,
    KNOWLEDGE_PAGES_FILE,
    RAG_ANSWER_EVAL_CASES_FILE,
    RAG_ANSWER_EVAL_REPORT_FILE,
)
from prokat24_ai_lab.rag.answer import answer_question
from prokat24_ai_lab.rag.openai_client import create_openai_generate_text
from prokat24_ai_lab.rag.ranker import create_hybrid_document_ranker


def main() -> None:
    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    document_embedding_index = load_document_embedding_index(DOCUMENT_EMBEDDING_INDEX_FILE)
    rag_eval_cases = load_rag_answer_eval_cases(RAG_ANSWER_EVAL_CASES_FILE)
    client = OpenAI()

    embed_text = create_openai_embed_text(client, document_embedding_index["model"])

    rank_documents = create_hybrid_document_ranker(
        pages=pages,
        document_embeddings=document_embedding_index["embeddings"],
        embed_text=embed_text,
    )

    generate_text = create_openai_generate_text(client)

    def rag_answerer(question: str) -> str:
        return answer_question(
            question=question,
            pages=pages,
            rank_documents=rank_documents,
            generate_text=generate_text,
            top_k=3,
        )

    results = run_rag_answer_evals(cases=rag_eval_cases, answerer=rag_answerer)

    created_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    git_commit = get_current_git_commit()
    git_dirty = is_git_worktree_dirty()

    report = build_rag_answer_eval_report(
        results,
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )

    save_rag_answer_eval_report(report, RAG_ANSWER_EVAL_REPORT_FILE)

    for result in results:
        print(f"Question: {result['question']}")
        print()
        print(f"Passed: {result['passed']}")
        print(f"Missing fragments: {result['missing_answer_fragments']}")
        print(f"Unexpected fragments: {result['unexpected_answer_fragments']}")
        print("Answer:")
        print(result["answer"])
        print()

    print(f"RAG answer evals: {report['passed_cases']} / {report['total_cases']}")
    print(f"Accuracy: {report['accuracy']:.2%}")
    print(f"Report saved to {RAG_ANSWER_EVAL_REPORT_FILE}")


if __name__ == "__main__":
    main()
