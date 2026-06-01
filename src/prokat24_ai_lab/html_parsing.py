from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

MAIN_CONTENT_SELECTORS = [
    "main.content .editor",
    ".contacts__info",
]


def extract_title(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    if soup.title is None or soup.title.string is None:
        return "No title found"

    return soup.title.string.strip()


def is_internal_link(url: str, allowed_domain: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc in ("", allowed_domain)


def extract_internal_links(html: str, base_url: str, allowed_domain: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        absolute_url = urljoin(base_url, href)

        if is_internal_link(absolute_url, allowed_domain):
            links.append(absolute_url)

    return sorted(set(links))


def extract_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["style", "script"]):
        tag.decompose()

    text = soup.get_text(separator="\n", strip=True)

    if not text:
        return "No text found"

    return text


def extract_main_content_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    for selector in MAIN_CONTENT_SELECTORS:
        content = soup.select_one(selector)

        if content is None:
            continue

        text = content.get_text(separator="\n", strip=True)

        if text:
            return text

    return "No main content found"
