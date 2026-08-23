import asyncio

from openai import AsyncOpenAI, OpenAI
from ragas.embeddings.base import embedding_factory
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    ContextRecall,
    Faithfulness,
)

from prokat24_ai_lab.embeddings.document_index import load_document_embedding_index
from prokat24_ai_lab.embeddings.openai_client import create_openai_embed_text
from prokat24_ai_lab.evals.ragas_evals import build_ragas_eval_result, load_ragas_eval_cases
from prokat24_ai_lab.knowledge.base import load_knowledge_base
from prokat24_ai_lab.project_paths import (
    DOCUMENT_EMBEDDING_INDEX_FILE,
    KNOWLEDGE_PAGES_FILE,
    RAGAS_EVAL_CASES_FILE,
)
from prokat24_ai_lab.rag.answer import answer_question_from_context_documents
from prokat24_ai_lab.rag.context import RetrievedContextDocument, build_retrieved_context_documents
from prokat24_ai_lab.rag.openai_client import create_openai_generate_text
from prokat24_ai_lab.rag.ranker import create_hybrid_document_ranker


def build_ragas_retrieved_contexts(
    context_documents: list[RetrievedContextDocument],
) -> list[str]:
    ragas_contexts: list[str] = []

    for document in context_documents:
        ragas_contexts.append(document["text"])

    return ragas_contexts


async def main() -> None:
    ragas_client = AsyncOpenAI()

    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=ragas_client)
    llm = llm_factory("gpt-4o-mini", client=ragas_client)

    faithfulness_scorer = Faithfulness(llm=llm)
    relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision_scorer = ContextPrecisionWithReference(llm=llm)
    context_recall_scorer = ContextRecall(llm=llm)

    cases = load_ragas_eval_cases(RAGAS_EVAL_CASES_FILE)
    case = cases[0]

    user_input = case["question"]
    reference = case["reference"]

    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    document_embedding_index = load_document_embedding_index(DOCUMENT_EMBEDDING_INDEX_FILE)
    retrieval_client = OpenAI()

    embed_text = create_openai_embed_text(retrieval_client, document_embedding_index["model"])

    rank_documents = create_hybrid_document_ranker(
        pages=pages,
        document_embeddings=document_embedding_index["embeddings"],
        embed_text=embed_text,
    )

    ranked_documents = rank_documents(user_input)

    context_documents = build_retrieved_context_documents(
        ranked_documents=ranked_documents,
        pages=pages,
        top_k=3,
    )

    generate_text = create_openai_generate_text(retrieval_client)

    response = answer_question_from_context_documents(
        question=user_input,
        context_documents=context_documents,
        generate_text=generate_text,
    )

    ragas_contexts = build_ragas_retrieved_contexts(context_documents)

    faithfulness_result = await faithfulness_scorer.ascore(
        user_input=user_input,
        response=response,
        retrieved_contexts=ragas_contexts,
    )

    relevancy_result = await relevancy_scorer.ascore(
        user_input=user_input,
        response=response,
    )

    context_precision_result = await context_precision_scorer.ascore(
        user_input=user_input,
        retrieved_contexts=ragas_contexts,
        reference=reference,
    )

    context_recall_result = await context_recall_scorer.ascore(
        user_input=user_input,
        retrieved_contexts=ragas_contexts,
        reference=reference,
    )

    result = build_ragas_eval_result(
        case=case,
        response=response,
        retrieved_contexts=ragas_contexts,
        faithfulness=faithfulness_result.value,
        answer_relevancy=relevancy_result.value,
        context_precision=context_precision_result.value,
        context_recall=context_recall_result.value,
    )

    print(f"Question: {user_input}")
    print(f"Reference: {reference}")
    print()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
