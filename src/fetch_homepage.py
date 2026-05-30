import requests
from bs4 import BeautifulSoup

from prokat24_ai_lab.prokat24_client import BASE_URL, fetch_page


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    if soup.title is None or soup.title.string is None:
        return "No title found"

    return soup.title.string.strip()


def main() -> None:
    html = fetch_page(BASE_URL)
    title = extract_title(html)

    print(f"URL: {BASE_URL}")
    print(f"HTML length: {len(html)} characters")
    print(f"Page title: {title}")


if __name__ == "__main__":
    main()
