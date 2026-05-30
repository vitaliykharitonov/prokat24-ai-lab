from prokat24_ai_lab.html_parsing import extract_title

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
