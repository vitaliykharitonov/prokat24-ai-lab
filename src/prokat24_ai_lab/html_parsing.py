from bs4 import BeautifulSoup


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    if soup.title is None or soup.title.string is None:
        return "No title found"

    return soup.title.string.strip()