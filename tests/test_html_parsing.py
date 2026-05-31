from prokat24_ai_lab.html_parsing import extract_title, extract_internal_links, is_internal_link, extract_text, extract_main_content_text


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


def test_extract_text_returns_visible_text() -> None:
    html = """
    <html>
        <body>
            <h1>Правила проката</h1>
            <p>Предоплата составляет 30%.</p>
        </body>
    </html>
    """

    result = extract_text(html)

    assert "Правила проката" in result
    assert "Предоплата составляет 30%" in result


def test_extract_text_removes_script_and_style_content() -> None:
    html = """
    <html>
        <head>
            <style>.hidden { color: red; }</style>
            <script>alert("x")</script>
        </head>
        <body>
            <h1>Правила проката</h1>
            <p>Предоплата составляет 30%.</p>
        </body>
    </html>
    """

    result = extract_text(html)

    assert "Правила проката" in result
    assert "Предоплата составляет 30%." in result
    assert "alert" not in result
    assert ".hidden" not in result


def test_extract_main_content_text_returns_editor_text_only() -> None:
    html = """
    <html>
        <body>
            <header>Меню сайту</header>
            <main class="content">
                <div class="container">
                    <h2 class="content__title">Про нас</h2>
                    <div class="editor">
                        <p>Прокат24 — комплексний сервіс.</p>
                        <p>Ми надаємо повний спектр послуг.</p>
                    </div>
                </div>
            </main>
            <footer>Футер сайту</footer>
        </body>
    </html>
    """

    result = extract_main_content_text(html)

    assert "Прокат24 — комплексний сервіс." in result
    assert "Ми надаємо повний спектр послуг." in result
    assert "Меню сайту" not in result
    assert "Футер сайту" not in result


def test_extract_main_content_text_returns_fallback_when_editor_is_missing() -> None:
    html = """
    <html>
        <body>
            <main class="content"></main>
        </body>
    </html>
    """

    result = extract_main_content_text(html)

    assert result == "No main content found"