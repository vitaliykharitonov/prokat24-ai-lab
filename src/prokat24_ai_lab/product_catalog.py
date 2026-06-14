from urllib.parse import urlparse

NON_PRODUCT_PATH_PARTS = {
    "pravila-prokata",
    "dostavka",
    "contact-us",
    "o-nas",
}


def is_product_category_url(url: str) -> bool:
    parsed = urlparse(url)
    path_parts = set(parsed.path.strip("/").split("/"))

    if parsed.netloc != "prokat24.com.ua":
        return False

    if not parsed.path.startswith("/ua/"):
        return False

    if not parsed.path.endswith("/"):
        return False

    return not path_parts & NON_PRODUCT_PATH_PARTS


def filter_product_category_urls(urls: list[str]) -> list[str]:
    filtered_urls = []

    for url in urls:
        if is_product_category_url(url):
            filtered_urls.append(url)

    return sorted(set(filtered_urls))
