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

```
python -m venv .venv
```

Activate virtual environment:

```
.\.venv\Scripts\Activate.ps1
```

Install project and dependencies:

```
pip install -e ".[dev]"
```

Run full project check:

```
python scripts/check_project.py
```

## Usage

Search the knowledge base:

```
python scripts/cli/search_knowledge_base.py "як отримати спорядження?"
```

Generate an answer from the knowledge base:

```
python scripts/cli/answer_question.py "де ви знаходитесь?"
```

Search the product catalog:

```
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

Run embedding search evals with the OpenAI API:

```powershell
python scripts/evals/run_embedding_search_evals.py
```

JSON reports are saved in the `reports/` directory.

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