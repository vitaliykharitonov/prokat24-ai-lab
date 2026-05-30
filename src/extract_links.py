from prokat24_ai_lab.prokat24_client import fetch_page
from prokat24_ai_lab.html_parsing import extract_internal_links
from prokat24_ai_lab.site_config import BASE_URL, ALLOWED_DOMAIN


def main() -> None:
    html = fetch_page(BASE_URL)
    links = extract_internal_links (html, BASE_URL, ALLOWED_DOMAIN)

    print(f"Found {len(links)} internal links")
    print()

    for link in links[:20]:
        print(link)


if __name__ == "__main__":
    main()
