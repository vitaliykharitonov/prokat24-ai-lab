from prokat24_ai_lab.html_parsing import extract_title
from prokat24_ai_lab.prokat24_client import fetch_page
from prokat24_ai_lab.site_config import BASE_URL


def main() -> None:
    html = fetch_page(BASE_URL)
    title = extract_title(html)

    print(f"URL: {BASE_URL}")
    print(f"HTML length: {len(html)} characters")
    print(f"Page title: {title}")


if __name__ == "__main__":
    main()
