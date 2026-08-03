# Prokat24 AI Lab

[![Check](https://github.com/vitaliykharitonov/prokat24-ai-lab/actions/workflows/check.yml/badge.svg)](https://github.com/vitaliykharitonov/prokat24-ai-lab/actions/workflows/check.yml)

Python AI/Evals project built around a real equipment rental website.

The project covers data collection, keyword and embedding search, rule-based answers, regression evals, and an automated
quality gate.

## Features

- Knowledge base and product catalog export
- Keyword search with normalization and aliases
- Product search and rule-based answers
- OpenAI embeddings and vector search
- Offline embedding evaluation
- JSON eval reports with Git metadata
- Automated checks with pytest, Ruff, mypy, and GitHub Actions

## Architecture

```text
src/prokat24_ai_lab/
├── knowledge/   # Knowledge base, search, and answers
├── products/    # Product catalog, search, and answers
├── embeddings/  # Embedding indexes and vector search
├── evals/       # Evaluation logic and reports
└── *.py         # Shared infrastructure and utilities
```

## Development setup

Create virtual environment:

```powershell
python -m venv .venv
```

Activate virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install project and dependencies:

```powershell
pip install -e ".[dev]"
```

Run full project check:

```powershell
python scripts/check_project.py
```

## Usage

Search the knowledge base:

```powershell
python scripts/cli/search_knowledge_base.py "як отримати спорядження?"
```

Generate an answer from the knowledge base:

```powershell
python scripts/cli/answer_question.py "де ви знаходитесь?"
```

Search the product catalog:

```powershell
python scripts/cli/search_products.py "скільки коштує 2 місний намет на тиждень"
```

## Data export

Export the knowledge base:

```powershell
python scripts/data/export_knowledge_base.py
```

Export the product catalog:

```powershell
python scripts/data/export_product_catalog.py
```

Generated JSON files are saved in the `data/` directory.

## Embeddings

Build document and query embedding indexes:

```powershell
python scripts/data/build_document_embedding_index.py
python scripts/data/build_query_embedding_index.py
```

These commands require an OpenAI API key.

## Evals

Run individual evaluation suites:

```powershell
python scripts/evals/run_search_evals.py
python scripts/evals/run_answer_evals.py
python scripts/evals/run_product_answer_evals.py
python scripts/evals/run_offline_embedding_search_evals.py
```

Run retrieval evals for a specific benchmark split:

```powershell
python scripts/evals/run_offline_embedding_search_evals.py --split dev
python scripts/evals/run_offline_embedding_search_evals.py --split holdout
python scripts/evals/run_hybrid_search_evals.py --split dev
python scripts/evals/run_hybrid_search_evals.py --split holdout
```

Run embedding search evals with the OpenAI API:

```powershell
python scripts/evals/run_embedding_search_evals.py
```

JSON reports are saved in the `reports/` directory.

## Retrieval benchmark

Retrieval Benchmark v2 contains 27 Ukrainian-language queries:

- 15 development cases used for regression checks and retrieval improvements
- 12 holdout cases used to measure generalization on previously unseen formulations

Each case includes an ID, category, difficulty level, and dataset split.

### Development split

| Retrieval method |           Top-1 |           Top-3 |
|------------------|----------------:|----------------:|
| Keyword search   |  14/15 (93.33%) | 15/15 (100.00%) |
| Embedding search |  12/15 (80.00%) | 15/15 (100.00%) |
| Hybrid RRF       | 15/15 (100.00%) | 15/15 (100.00%) |

### Holdout split

| Retrieval method |         Top-1 |          Top-3 |
|------------------|--------------:|---------------:|
| Keyword search   | 5/12 (41.67%) |  8/12 (66.67%) |
| Embedding search | 6/12 (50.00%) | 10/12 (83.33%) |
| Hybrid RRF       | 7/12 (58.33%) |  9/12 (75.00%) |

The development split is part of the automated quality gate. Its established
retrieval metrics are protected against regressions.

The holdout split is evaluated separately and does not block regular development.
Its results represent the first observed generalization baseline and should not be
used for direct tuning against individual queries.

Running either retrieval script without `--split` evaluates all benchmark cases.
The automated project check runs retrieval quality gates only against the stable
development split.

## Development workflow

1. Create a feature branch.
2. Make code and test changes.
3. Run the full project check:

```powershell
python scripts/check_project.py
```

4. Commit and push the changes.
5. Open a pull request.
6. Merge after the GitHub Actions check passes.