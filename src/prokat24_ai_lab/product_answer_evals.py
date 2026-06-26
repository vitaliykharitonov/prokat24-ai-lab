from typing import TypedDict

from prokat24_ai_lab.product_answers import answer_product_query
from prokat24_ai_lab.product_catalog import Product


class ProductAnswerEvalCase(TypedDict):
    query: str
    expected_fragments: list[str]


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
            "Deposit: 4000 грн",
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
]


def product_answer_eval_passed(products: list[Product], eval_case: ProductAnswerEvalCase) -> bool:
    answer = answer_product_query(products, eval_case["query"])

    return all(fragment in answer for fragment in eval_case["expected_fragments"])


def count_passed_product_answer_evals(products: list[Product]) -> int:
    passed_count = 0

    for eval_case in PRODUCT_ANSWER_EVAL_CASES:
        if product_answer_eval_passed(products, eval_case):
            passed_count += 1

    return passed_count
