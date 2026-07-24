from prokat24_ai_lab.html_parsing import extract_internal_links
from prokat24_ai_lab.product_catalog import filter_product_category_urls
from prokat24_ai_lab.prokat24_client import fetch_page
from prokat24_ai_lab.site_config import ALLOWED_DOMAIN, BASE_URL


def main() -> None:
    html = fetch_page(BASE_URL)
    extracted_links = extract_internal_links(html, BASE_URL, ALLOWED_DOMAIN)
    filtered_links = filter_product_category_urls(extracted_links)

    print(f"Found {len(filtered_links)} product category URLs")

    for link in filtered_links:
        print(link)


if __name__ == "__main__":
    main()
