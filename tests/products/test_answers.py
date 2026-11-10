from prokat24_ai_lab.products.answers import answer_product_query
from prokat24_ai_lab.products.catalog import Product


def test_answer_product_query_returns_formatted_search_results(
    two_person_tent: Product, three_person_tent: Product
) -> None:
    product_list = [two_person_tent, three_person_tent]
    query = "2 місних"

    result = answer_product_query(product_list, query, limit=5)

    assert "Прокат 2-місних наметів" in result
    assert "Прокат 3-місних наметів" not in result


def test_answer_product_query_returns_not_found_message_when_no_products_match(
    two_person_tent: Product,
) -> None:
    product_list = [two_person_tent]
    query = "абракадабра"

    result = answer_product_query(product_list, query)

    assert result == "Товари не знайдено"


def test_answer_product_query_returns_specific_period_price_when_period_detected(
    two_person_tent: Product, three_person_tent: Product
) -> None:
    product_list = [two_person_tent, three_person_tent]
    query = "скільки коштує 2 місний намет на тиждень"

    result = answer_product_query(product_list, query)

    assert "Прокат 2-місних наметів" in result
    assert "Тиждень: 1200 грн" in result
    assert "Застава:" in result
    assert two_person_tent["url"] in result


def test_answer_product_query_returns_deposit_when_deposit_intent_detected(
    two_person_tent: Product, three_person_tent: Product
) -> None:
    product_list = [two_person_tent, three_person_tent]
    query = "яка застава за 2 місний намет"

    result = answer_product_query(product_list, query)

    assert "Прокат 2-місних наметів" in result
    assert "Застава: 4000 грн або Закордонний паспорт/права" in result
    assert two_person_tent["url"] in result


def test_answer_product_query_returns_product_details(full_product: Product) -> None:
    query = "які характеристики у 2 місний намет"

    result = answer_product_query([full_product], query)

    assert "Прокат 2-місних наметів" in result
    assert "Прокат легких двомісних наметів" in result
    assert "Характеристики:" in result
    assert "Розмір: 205 х 240 см" in result
    assert "Вага: 3.31 кг" in result
    assert "Комплектація: Намет, зовнішній тент" in result


def test_answer_product_query_returns_requested_attribute(full_product: Product) -> None:
    query = "яка вага 2 місний намет"

    result = answer_product_query([full_product], query)

    assert "Прокат 2-місних наметів" in result
    assert "Вага: 3.31 кг" in result
    assert "Характеристики:" not in result


def test_answer_product_query_returns_message_when_requested_attribute_missing(
    two_person_tent: Product,
) -> None:
    query = "яка вага 2 місний намет"

    result = answer_product_query([two_person_tent], query)

    assert "Прокат 2-місних наметів" in result
    assert "Характеристику Вага не знайдено" in result
