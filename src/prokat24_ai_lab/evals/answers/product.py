import datetime
import json
from pathlib import Path
from typing import TypedDict

from prokat24_ai_lab.evals.report_metadata import build_eval_report_metadata
from prokat24_ai_lab.products.answers import answer_product_query
from prokat24_ai_lab.products.catalog import Product


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
    created_at: str
    git_commit: str | None
    git_dirty: bool | None
    passed: bool
    passed_count: int
    total_count: int
    failed_count: int
    failures: list[ProductAnswerEvalFailure]


def load_product_answer_eval_cases(input_file: Path) -> list[ProductAnswerEvalCase]:
    with open(input_file, encoding="utf-8") as file:
        loaded_data: list[ProductAnswerEvalCase] = json.load(file)

    return loaded_data


def product_answer_eval_passed(products: list[Product], eval_case: ProductAnswerEvalCase) -> bool:
    answer = answer_product_query(products, eval_case["query"])

    return all(fragment in answer for fragment in eval_case["expected_fragments"])


def find_missing_fragments(answer: str, expected_fragments: list[str]) -> list[str]:
    missing_fragments = []

    for fragment in expected_fragments:
        if fragment not in answer:
            missing_fragments.append(fragment)

    return missing_fragments


def find_failed_product_answer_evals(
    products: list[Product],
    cases: list[ProductAnswerEvalCase],
) -> list[ProductAnswerEvalFailure]:
    failures: list[ProductAnswerEvalFailure] = []

    for eval_case in cases:
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
    passed_count: int,
    total_count: int,
    failures: list[ProductAnswerEvalFailure],
    created_at: str | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> ProductAnswerEvalReport:
    if created_at is None:
        created_at = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")

    metadata = build_eval_report_metadata(
        report_type="product_answer_eval_report",
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    failed_count = total_count - passed_count
    passed = failed_count == 0

    return {
        **metadata,
        "passed": passed,
        "passed_count": passed_count,
        "total_count": total_count,
        "failed_count": failed_count,
        "failures": failures,
    }


def build_product_answer_eval_report_for_products(
    products: list[Product],
    cases: list[ProductAnswerEvalCase],
    created_at: str | None = None,
    git_commit: str | None = None,
    git_dirty: bool | None = None,
) -> ProductAnswerEvalReport:
    failures = find_failed_product_answer_evals(products, cases)
    total_count = len(cases)
    passed_count = total_count - len(failures)

    return build_product_answer_eval_report(
        passed_count,
        total_count,
        failures,
        created_at=created_at,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )


def save_product_answer_eval_report(report: ProductAnswerEvalReport, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
