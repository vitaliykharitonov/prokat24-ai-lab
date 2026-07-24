from prokat24_ai_lab.html_parsing import extract_internal_links
from prokat24_ai_lab.products.catalog import (
    build_product_catalog,
    filter_leaf_product_category_urls,
    filter_product_category_urls,
    save_products,
)
from prokat24_ai_lab.project_paths import PRODUCTS_FILE
from prokat24_ai_lab.prokat24_client import fetch_page
from prokat24_ai_lab.site_config import ALLOWED_DOMAIN, BASE_URL


def main() -> None:
    html = fetch_page(BASE_URL)
    links = extract_internal_links(html, BASE_URL, ALLOWED_DOMAIN)
    category_urls = filter_product_category_urls(links)
    leaf_category_urls = filter_leaf_product_category_urls(category_urls)

    print(f"Found {len(category_urls)} product categories")
    print(f"Found {len(leaf_category_urls)} leaf product categories")

    products = build_product_catalog(leaf_category_urls)

    print(f"Found {len(products)} products")

    save_products(products, PRODUCTS_FILE)

    print(f"Saved products to {PRODUCTS_FILE}")


if __name__ == "__main__":
    main()
