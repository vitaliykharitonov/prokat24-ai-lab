import re


def normalize_text(text: str) -> str:
    words = re.findall(r"[0-9a-zа-яіїєґʼ']+", text.lower())

    return " ".join(words)
