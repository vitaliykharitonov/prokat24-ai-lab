from pathlib import Path


def find_project_root(start_path: Path) -> Path:
    current_path = start_path.resolve()

    if current_path.is_file():
        current_path = current_path.parent

    for parent in [current_path, *current_path.parents]:
        if (parent / ".git").exists():
            return parent

    raise FileNotFoundError("Project root was not found")


PROJECT_ROOT = find_project_root(Path(__file__))
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

KNOWLEDGE_PAGES_FILE = DATA_DIR / "knowledge_pages.json"
SEARCH_EVAL_CASES_FILE = DATA_DIR / "search_eval_cases.json"
ANSWER_EVAL_CASES_FILE = DATA_DIR / "answer_eval_cases.json"
PRODUCTS_FILE = DATA_DIR / "products.json"

SEARCH_EVAL_REPORT_FILE = REPORTS_DIR / "search_eval_report.json"
ANSWER_EVAL_REPORT_FILE = REPORTS_DIR / "answer_eval_report.json"
