# Grader Comparison Baseline

This document records the baseline comparison of three answer-evaluation approaches:

- deterministic grader
- holistic LLM judge
- criterion-level LLM judge

The development dataset was used during grader and prompt development.
The holdout dataset was evaluated only after the grading architecture was finalized.

## Development Dataset

Cases: 11

| Grader        |        Accuracy |
|---------------|----------------:|
| Deterministic |   2/11 (18.18%) |
| Holistic LLM  |   9/11 (81.82%) |
| Criterion LLM | 11/11 (100.00%) |

The main weakness of the holistic LLM judge was the `irrelevant_extra` category.
It produced stable false positives for two cases even though the answers contained information that was not necessary to
answer the user's question.

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

## Stability

Each holdout case was evaluated three times by both LLM-based graders.

The criterion-level LLM judge produced the same correct verdict in all three runs for all 12 holdout cases.

The holistic LLM judge was unstable on:

`address_irrelevant_extra_01`

Expected verdict: `FAIL`

Results:

- Run 1: PASS
- Run 2: PASS
- Run 3: FAIL

Match rate: 33.33%

## Key Findings

Deterministic fragment-based grading works well for simple literal checks but does not reliably evaluate semantic
properties such as paraphrasing, negation, unsupported claims, completeness, or relevance.

The holistic LLM judge handles semantic evaluation substantially better than the deterministic grader, but it can
confuse grounded information with relevant information and can produce unstable verdicts.

Criterion-level grading separates individual rubric decisions such as correctness, relevance, groundedness, and
completeness. The final overall verdict is then aggregated deterministically in Python.

On the current holdout dataset, this architecture achieved:

- 100% accuracy
- 100% precision
- 100% recall
- consistent verdicts across three runs

## Limitations

The holdout dataset contains only 12 cases.

Therefore, the current result does not demonstrate that criterion-level grading has 100% accuracy in general.

The benchmark only shows that the criterion-level architecture generalized successfully from the development dataset to
the current unseen holdout dataset.

Future evaluation should use larger and more diverse datasets, including product questions and additional RAG failure
modes.