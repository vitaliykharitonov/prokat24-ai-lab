import requests
from bs4 import BeautifulSoup


URL = "https://prokat24.com.ua/"


def fetch_page(url: str) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    if soup.title is None or soup.title.string is None:
        return "No title found"

    return soup.title.string.strip()


def main() -> None:
    html = fetch_page(URL)
    title = extract_title(html)

    print(f"URL: {URL}")
    print(f"HTML length: {len(html)} characters")
    print(f"Page title: {title}")


if __name__ == "__main__":
    main()
