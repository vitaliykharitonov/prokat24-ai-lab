import json
from pathlib import Path
from typing import TypedDict

from prokat24_ai_lab.product_answers import answer_product_query
from prokat24_ai_lab.product_catalog import Product


class ProductAnswerEvalCase(TypedDict):
    query: str
    expected_fragments: list[str]


class ProductAnswerEvalFailure(TypedDict):
    query: str
    answer: str
    missing_fragments: list[str]


class ProductAnswerEvalReport(TypedDict):
    report_type: str
    schema_version: int
    passed: bool
    passed_count: int
    total_count: int
    failed_count: int
    failures: list[ProductAnswerEvalFailure]


PRODUCT_ANSWER_EVAL_CASES: list[ProductAnswerEvalCase] = [
    {
        "query": "скільки коштує 2 місний намет на тиждень",
        "expected_fragments": [
            "Прокат 2-місних наметів",
            "Тиждень: 1200 грн",
        ],
    },
    {
        "query": "скільки коштує 2 місний намет на 2 тижні",
        "expected_fragments": [
            "Прокат 2-місних наметів",
            "2 тижні: 2300 грн",
        ],
    },
    {
        "query": "яка застава за 2 місний намет",
        "expected_fragments": [
            "Прокат 2-місних наметів",
            "Застава: 4000 грн",
        ],
    },
    {
        "query": "прокат спальника на 2 тижні",
        "expected_fragments": [
            "спальників",
            "2 тижні:",
        ],
    },
    {
        "query": "прокат човна на тиждень",
        "expected_fragments": [
            "човн",
            "Тиждень:",
        ],
    },
    {
        "query": "скільки коштує спальник на тиждень",
        "expected_fragments": [
            "Прокат осінніх спальників Asleep",
            "Тиждень: 650 грн",
        ],
    },
    {
        "query": "яка застава за спальник",
        "expected_fragments": [
            "Прокат осінніх спальників Asleep",
            "Застава: 2000 грн",
        ],
    },
    {
        "query": "прокат човна",
        "expected_fragments": [
            "Прокат 2-місних човнів",
            "Ціни:",
            "Тиждень: 2000 грн",
        ],
    },
    {
        "query": "скільки коштує 2 місний човен на тиждень",
        "expected_fragments": [
            "Прокат 2-місних човнів",
            "Тиждень: 2000 грн",
        ],
    },
    {
        "query": "скільки коштує 3 місний човен на тиждень",
        "expected_fragments": [
            "Прокат 3-місних човнів",
            "Тиждень: 2700 грн",
        ],
    },
    {
        "query": "яка застава за 2 місний човен",
        "expected_fragments": [
            "Прокат 2-місних човнів",
            "Застава: 8000 грн",
        ],
    },
    {
        "query": "яка застава за 3 місний човен",
        "expected_fragments": [
            "Прокат 3-місних човнів",
            "Застава: 10000 грн",
        ],
    },
    {
        "query": "скільки коштує 2 місний намет на 5 днів",
        "expected_fragments": [
            "Прокат 2-місних наметів",
            "5 днів: 980 грн",
        ],
    },
    {
        "query": "скільки коштує спальник на 5 днів",
        "expected_fragments": [
            "Прокат осінніх спальників Asleep",
            "5 днів: 520 грн",
        ],
    },
    {
        "query": "скільки коштує 2 місний човен на 5 днів",
        "expected_fragments": [
            "Прокат 2-місних човнів",
            "5 днів: 1550 грн",
        ],
    },
    {
        "query": "скільки буде коштувати на 5 днів 2 місний намет",
        "expected_fragments": [
            "Прокат 2-місних наметів",
            "5 днів: 980 грн",
        ],
    },
    {
        "query": "чи є в наявності 2 місний намет",
        "expected_fragments": [
            "Прокат 2-місних наметів",
            "Актуальну наявність краще уточнити у менеджера",
        ],
    },
    {
        "query": "які характеристики у 2 місний намет",
        "expected_fragments": [
            "Прокат 2-місних наметів",
            "Прокат легких двомісних наметів",
            "Характеристики:",
            "Розмір: 205 х 240 см",
            "Вага: 3.31 кг",
            "Комплектація: Намет, зовнішній тент",
        ],
    },
    {
        "query": "яка вага 2 місний намет",
        "expected_fragments": [
            "Прокат 2-місних наметів",
            "Вага: 3.31 кг",
        ],
    },
    {
        "query": "якого розміру 2 місний намет",
        "expected_fragments": [
            "Прокат 2-місних наметів",
            "Розмір: 205 х 240 см",
        ],
    },
    {
        "query": "2 місний намет комплектація",
        "expected_fragments": [
            "Прокат 2-місних наметів",
            "Комплектація: Намет, зовнішній тент, набір кілочків (10 шт), дуги (2 шт), чохли",
        ],
    },
    {
        "query": "яка вага 2 місного намету",
        "expected_fragments": [
            "Прокат 2-місних наметів",
            "Вага: 3.31 кг",
        ],
    },
    {
        "query": "яка вага лижних черевиків",
        "expected_fragments": [
            "Прокат лижних черевиків",
            "Характеристику Вага не знайдено",
        ],
    },
]


def product_answer_eval_passed(products: list[Product], eval_case: ProductAnswerEvalCase) -> bool:
    answer = answer_product_query(products, eval_case["query"])

    return all(fragment in answer for fragment in eval_case["expected_fragments"])


def find_missing_fragments(answer: str, expected_fragments: list[str]) -> list[str]:
    missing_fragments = []

    for fragment in expected_fragments:
        if fragment not in answer:
            missing_fragments.append(fragment)

    return missing_fragments


def find_failed_product_answer_evals(products: list[Product]) -> list[ProductAnswerEvalFailure]:
    failures: list[ProductAnswerEvalFailure] = []

    for eval_case in PRODUCT_ANSWER_EVAL_CASES:
        query = eval_case["query"]
        answer = answer_product_query(products, query)
        missing_fragments = find_missing_fragments(answer, eval_case["expected_fragments"])

        if missing_fragments:
            failure: ProductAnswerEvalFailure = {
                "query": query,
                "answer": answer,
                "missing_fragments": missing_fragments,
            }

            failures.append(failure)

    return failures


def build_product_answer_eval_report(
    passed_count: int, total_count: int, failures: list[ProductAnswerEvalFailure]
) -> ProductAnswerEvalReport:
    failed_count = total_count - passed_count
    passed = failed_count == 0

    return {
        "report_type": "product-answer-eval",
        "schema_version": 1,
        "passed": passed,
        "passed_count": passed_count,
        "total_count": total_count,
        "failed_count": failed_count,
        "failures": failures,
    }


def build_product_answer_eval_report_for_products(
    products: list[Product],
) -> ProductAnswerEvalReport:
    failures = find_failed_product_answer_evals(products)
    total_count = len(PRODUCT_ANSWER_EVAL_CASES)
    passed_count = total_count - len(failures)

    return build_product_answer_eval_report(passed_count, total_count, failures)


def save_product_answer_eval_report(report: ProductAnswerEvalReport, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
