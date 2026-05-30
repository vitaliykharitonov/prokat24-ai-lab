from prokat24_ai_lab.html_parsing import extract_title, extract_internal_links, is_internal_link


def test_extract_title_returns_title_text() -> None:
    html = """
    <html>
        <head>
            <title> ПРОКАТ24 </title>
        </head>
        <body></body>
    </html>
    """

    result = extract_title(html)
    assert result == "ПРОКАТ24"


def test_extract_title_returns_fallback_when_title_is_missing() -> None:
    html = """
    <html>
        <head></head>
        <body></body>
    </html>
    """

    result = extract_title(html)
    assert result == "No title found"


def test_is_internal_returns_true_for_same_domain() -> None:
    result = is_internal_link("https://prokat24.com.ua/ua/dostavka/", "prokat24.com.ua")

    assert result is True


def test_is_internal_link_returns_false_for_external_domain() -> None:
    result = is_internal_link("https://google.com/maps", "prokat24.com.ua")

    assert result is False


def test_extract_internal_links_returns_unique_absolute_internal_links() -> None:
    html = """
    <html>
        <body>
            <a href="/ua/dostavka/">Доставка</a>
            <a href="https://prokat24.com.ua/ua/contact-us/">Контакты</a>
            <a href="https://google.com/maps">Google Maps</a>
            <a href="/ua/dostavka/">Доставка duplicate</a>
        </body>
    </html>
    """

    result = extract_internal_links(
        html=html,
        base_url="https://prokat24.com.ua",
        allowed_domain="prokat24.com.ua",
    )

    assert result == [
        "https://prokat24.com.ua/ua/contact-us/",
        "https://prokat24.com.ua/ua/dostavka/",
    ]