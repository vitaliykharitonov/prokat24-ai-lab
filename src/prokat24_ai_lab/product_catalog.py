from urllib.parse import urlparse

PRODUCT_CATEGORY_ROOT_PATHS = {
    "/ua/gornolyzhnoe-snaryazhenie/",
    "/ua/prokat-mebeli/",
    "/ua/prokat-razvlechenij/",
    "/ua/rybolovnoe-snaryazhenie/",
    "/ua/turisticheskoe-snaryazhenie/",
    "/ua/velosipedy-i-roliki/",
}


def is_product_category_url(url: str) -> bool:
    parsed = urlparse(url)

    if parsed.netloc != "prokat24.com.ua":
        return False

    if not parsed.path.endswith("/"):
        return False

    return any(
        parsed.path.startswith(root_path)
        for root_path in PRODUCT_CATEGORY_ROOT_PATHS
    )


def filter_product_category_urls(urls: list[str]) -> list[str]:
    filtered_urls = []

    for url in urls:
        if is_product_category_url(url):
            filtered_urls.append(url)

    return sorted(set(filtered_urls))
