from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from prokat24_client import BASE_URL, fetch_page


def is_internal_link(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.netloc in ("", "prokat24.com.ua")


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: list[str] = []

    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        absolute_url = urljoin(base_url, href)

        if is_internal_link(absolute_url):
            links.append(absolute_url)

    return sorted(set(links))


def main() -> None:
    html = fetch_page(BASE_URL)
    links = extract_links(html, BASE_URL)

    print(f"Found {len(links)} internal links")
    print()

    for link in links[:20]:
        print(link)


if __name__ == "__main__":
    main()
