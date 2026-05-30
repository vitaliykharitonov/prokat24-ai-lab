import requests


BASE_URL = "https://prokat24.com.ua/"


def fetch_page(url: str) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.text