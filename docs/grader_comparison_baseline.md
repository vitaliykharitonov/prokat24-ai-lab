# Grader Comparison Baseline

This document compares four answer-evaluation approaches:

- deterministic grader
- holistic LLM judge
- criterion-level LLM judge
- DeepEval criterion-level GEval grader

The development dataset was used for grader calibration.
The holdout dataset was evaluated only after the grader configuration was finalized.

## Development Dataset

Cases: 11

| Grader        |        Accuracy |
|---------------|----------------:|
| Deterministic |   2/11 (18.18%) |
| Holistic LLM  |   9/11 (81.82%) |
| Criterion LLM | 11/11 (100.00%) |

The main weakness of the holistic LLM judge was `irrelevant_extra`: it sometimes treated grounded but unnecessary
information as relevant.

## Holdout Dataset

Cases: 12

| Grader        |        Accuracy | Precision |  Recall |
|---------------|----------------:|----------:|--------:|
| Deterministic |   4/12 (33.33%) |    28.57% |  40.00% |
| Holistic LLM  |  11/12 (91.67%) |    83.33% | 100.00% |
| Criterion LLM | 12/12 (100.00%) |   100.00% | 100.00% |

### Accuracy by category

| Category          | Deterministic | Holistic LLM | Criterion LLM |
|-------------------|--------------:|-------------:|--------------:|
| correct           |           2/2 |          2/2 |           2/2 |
| incorrect         |           2/2 |          2/2 |           2/2 |
| paraphrase        |           0/2 |          2/2 |           2/2 |
| negation          |           0/1 |          1/1 |           1/1 |
| unsupported_claim |           0/1 |          1/1 |           1/1 |
| short_correct     |           0/1 |          1/1 |           1/1 |
| partial_answer    |           0/1 |          1/1 |           1/1 |
| irrelevant_extra  |           0/2 |          1/2 |           2/2 |

## DeepEval Comparison

DeepEval was integrated by converting each rubric criterion into a separate `GEval` metric and aggregating criterion
verdicts deterministically.

### Development

Initial result:

- 9/11 matched (81.82%)

Calibration on the development set:

- groundedness threshold: `0.9`
- relevance threshold: `0.9`
- stricter relevance evaluation steps

After calibration:

- 11/11 matched (100.00%)

### Holdout

The calibrated DeepEval grader was frozen before holdout evaluation.

Result:

- 12/12 matched (100.00%)

No further calibration was performed on the holdout dataset.

The DeepEval grader uses `gpt-5.4`, while the custom criterion-level judge uses `gpt-5.4-mini-2026-03-17`, so this is
not a strict apples-to-apples architecture comparison.

## Stability

The custom LLM graders were evaluated across three runs.

The criterion-level judge produced correct and stable verdicts for all 12 holdout cases.

The holistic LLM judge was unstable on `address_irrelevant_extra_01`:

- expected: FAIL
- runs: PASS, PASS, FAIL
- match rate: 33.33%

## Key Findings

Deterministic fragment checks work for simple literal cases but fail on semantic properties such as paraphrasing,
negation, unsupported claims, completeness, and relevance.

Holistic LLM grading handles semantics better but can confuse groundedness with relevance.

Criterion-level grading separates rubric decisions and aggregates them deterministically, which performed better on the
current datasets.

The calibrated DeepEval criterion-level grader also matched all 12 holdout human labels.

## Limitations

The datasets are small:

- development: 11 cases
- holdout: 12 cases

The current results do not demonstrate general 100% grader accuracy.

Future evaluation should use larger and more diverse datasets, including product questions and additional RAG failure
modes.