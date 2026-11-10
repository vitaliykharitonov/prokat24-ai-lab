import pytest

from prokat24_ai_lab.products.catalog import Product


@pytest.fixture
def two_person_tent() -> Product:
    return {
        "url": "https://prokat24.com.ua/ua/test-tent/",
        "name": "Прокат 2-місних наметів",
        "description": "Опис 2-місного намету.",
        "rental_prices": {
            "1-й день": "250 грн",
            "Тиждень": "1200 грн",
        },
        "deposit": "4000 грн або Закордонний паспорт/права",
        "attributes": {
            "К-сть осіб": "2 - 3",
        },
    }


@pytest.fixture
def three_person_tent() -> Product:
    return {
        "url": "https://prokat24.com.ua/ua/test-tent3/",
        "name": "Прокат 3-місних наметів",
        "description": "Опис 3-місного намету.",
        "rental_prices": {
            "1-й день": "300 грн",
            "Тиждень": "1500 грн",
        },
        "deposit": "4500 грн або закордонний паспорт/права + 1000 грн",
        "attributes": {
            "К-сть осіб": "3 - 4",
        },
    }


@pytest.fixture
def full_product() -> Product:
    return {
        "url": "https://prokat24.com.ua/ua/turisticheskoe-snaryazhenie/prokat-palatok/prokat-palatki-2-mestnaya/",
        "name": "Прокат 2-місних наметів",
        "description": (
            "Прокат легких двомісних наметів Terra Incognita Alfa2. "
            "Ідеально підходить для туристичних походів. "
            "Ефективна вентиляція, просте та швидке встановлення."
        ),
        "rental_prices": {
            "1-й день": "250 грн",
            "2-й день": "220 грн",
            "3-й день": "190 грн",
            "з 4-го дня": "160 грн",
            "Тиждень": "1200 грн",
            "2 тижні": "2300 грн",
        },
        "deposit": "4000 грн або Закордонний паспорт/права",
        "attributes": {
            "Розмір": "205 х 240 см",
            "Висота": "115 см",
            "Вага": "3.31 кг",
            "К-сть осіб": "2 - 3",
            "Колір": "Хакі, пісочний",
            "Габарити упаковки": "15 х 15 х 45 см",
            "Комплектація": ("Намет, зовнішній тент, набір кілочків (10 шт), дуги (2 шт), чохли"),
        },
    }


@pytest.fixture
def product_without_extra_day_price() -> Product:
    return {
        "url": "https://prokat24.com.ua/ua/test-tent/",
        "name": "Прокат 2-місних наметів",
        "description": "Опис 2-місного намету.",
        "rental_prices": {
            "1-й день": "250 грн",
            "2-й день": "220 грн",
            "3-й день": "190 грн",
            "Тиждень": "1200 грн",
            "2 тижні": "2300 грн",
        },
        "deposit": "4000 грн або Закордонний паспорт/права",
        "attributes": {
            "К-сть осіб": "2 - 3",
        },
    }
