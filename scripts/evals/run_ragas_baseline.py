import asyncio
from argparse import ArgumentParser, Namespace

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
from prokat24_ai_lab.evals.answers.ragas import (
    RagasEvalCase,
    RagasEvalResult,
    build_ragas_eval_result,
    load_ragas_eval_cases,
    summarize_ragas_eval_results,
)
from prokat24_ai_lab.knowledge.base import KnowledgePage, load_knowledge_base
from prokat24_ai_lab.project_paths import (
    DOCUMENT_EMBEDDING_INDEX_FILE,
    KNOWLEDGE_PAGES_FILE,
    RAGAS_EVAL_CASES_FILE,
)
from prokat24_ai_lab.rag.answer import DocumentRanker, answer_question_from_context_documents
from prokat24_ai_lab.rag.context import RetrievedContextDocument, build_retrieved_context_documents
from prokat24_ai_lab.rag.generation import TextGenerator
from prokat24_ai_lab.rag.openai_client import create_openai_generate_text
from prokat24_ai_lab.rag.ranker import create_hybrid_document_ranker


def build_ragas_retrieved_contexts(
    context_documents: list[RetrievedContextDocument],
) -> list[str]:
    ragas_contexts: list[str] = []

    for document in context_documents:
        ragas_contexts.append(document["text"])

    return ragas_contexts


async def evaluate_ragas_case(
    case: RagasEvalCase,
    pages: list[KnowledgePage],
    rank_documents: DocumentRanker,
    generate_text: TextGenerator,
    faithfulness_scorer: Faithfulness,
    relevancy_scorer: AnswerRelevancy,
    context_precision_scorer: ContextPrecisionWithReference,
    context_recall_scorer: ContextRecall,
    metric: str | None,
) -> RagasEvalResult:
    user_input = case["question"]
    reference = case["reference"]

    ranked_documents = rank_documents(user_input)

    context_documents = build_retrieved_context_documents(
        ranked_documents=ranked_documents,
        pages=pages,
        top_k=3,
    )

    response = answer_question_from_context_documents(
        question=user_input,
        context_documents=context_documents,
        generate_text=generate_text,
    )

    ragas_contexts = build_ragas_retrieved_contexts(context_documents)

    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None

    if metric is None or metric == "faithfulness":
        faithfulness_result = await faithfulness_scorer.ascore(
            user_input=user_input,
            response=response,
            retrieved_contexts=ragas_contexts,
        )

        faithfulness = faithfulness_result.value

    if metric is None or metric == "answer-relevancy":
        relevancy_result = await relevancy_scorer.ascore(
            user_input=user_input,
            response=response,
        )

        answer_relevancy = relevancy_result.value

    if metric is None or metric == "context-precision":
        context_precision_result = await context_precision_scorer.ascore(
            user_input=user_input,
            retrieved_contexts=ragas_contexts,
            reference=reference,
        )

        context_precision = context_precision_result.value

    if metric is None or metric == "context-recall":
        context_recall_result = await context_recall_scorer.ascore(
            user_input=user_input,
            retrieved_contexts=ragas_contexts,
            reference=reference,
        )

        context_recall = context_recall_result.value

    result = build_ragas_eval_result(
        case=case,
        response=response,
        retrieved_contexts=ragas_contexts,
        faithfulness=faithfulness,
        answer_relevancy=answer_relevancy,
        context_precision=context_precision,
        context_recall=context_recall,
    )

    return result


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Run Ragas on one or all cases")
    parser.add_argument(
        "--case-index",
        type=int,
        help="Enter case index to evaluate",
        default=None,
    )
    parser.add_argument(
        "--metric",
        choices=(
            "faithfulness",
            "answer-relevancy",
            "context-precision",
            "context-recall",
        ),
        default=None,
        help="Run only one Ragas metric",
    )

    return parser.parse_args()


def format_metric_value(value: float | None) -> str:
    if value is None:
        return "N/A"

    return f"{value:.3f}"


async def main() -> None:
    args = parse_args()

    cases = load_ragas_eval_cases(RAGAS_EVAL_CASES_FILE)

    if args.case_index is not None:
        if args.case_index < 0 or args.case_index >= len(cases):
            raise IndexError("Case index out of range")

        cases = [cases[args.case_index]]

    ragas_client = AsyncOpenAI()

    embeddings = embedding_factory("openai", model="text-embedding-3-small", client=ragas_client)
    llm = llm_factory("gpt-4o-mini", client=ragas_client)

    faithfulness_scorer = Faithfulness(llm=llm)
    relevancy_scorer = AnswerRelevancy(llm=llm, embeddings=embeddings)
    context_precision_scorer = ContextPrecisionWithReference(llm=llm)
    context_recall_scorer = ContextRecall(llm=llm)

    pages = load_knowledge_base(KNOWLEDGE_PAGES_FILE)
    document_embedding_index = load_document_embedding_index(DOCUMENT_EMBEDDING_INDEX_FILE)
    retrieval_client = OpenAI()

    embed_text = create_openai_embed_text(retrieval_client, document_embedding_index["model"])

    rank_documents = create_hybrid_document_ranker(
        pages=pages,
        document_embeddings=document_embedding_index["embeddings"],
        embed_text=embed_text,
    )

    generate_text = create_openai_generate_text(retrieval_client)

    results: list[RagasEvalResult] = []

    for case in cases:
        result = await evaluate_ragas_case(
            case=case,
            pages=pages,
            rank_documents=rank_documents,
            generate_text=generate_text,
            faithfulness_scorer=faithfulness_scorer,
            relevancy_scorer=relevancy_scorer,
            context_precision_scorer=context_precision_scorer,
            context_recall_scorer=context_recall_scorer,
            metric=args.metric,
        )
        results.append(result)

    summary = summarize_ragas_eval_results(results)

    for result in results:
        print(f"Question: {result['question']}")
        print("Response:")
        print(result["response"])
        print(f"Faithfulness: {result['faithfulness']}")
        print(f"Answer relevancy: {result['answer_relevancy']}")
        print(f"Context precision: {result['context_precision']}")
        print(f"Context recall: {result['context_recall']}")
        print()

    print("RAGAS SUMMARY")
    print(f"Cases: {summary['total_cases']}")
    print(f"Average faithfulness: {format_metric_value(summary['average_faithfulness'])}")
    print(f"Average answer-relevancy: {format_metric_value(summary['average_answer_relevancy'])}")
    print(f"Average context precision: {format_metric_value(summary['average_context_precision'])}")
    print(f"Average context recall: {format_metric_value(summary['average_context_recall'])}")


if __name__ == "__main__":
    asyncio.run(main())
