product = {
    "category": "Туристическое снаряжение",
    "name": "Палатка 4-местная",
    "price_per_day": 400,
    "days": 5,
    "deposit": 6000,
    "is_available": True,
}

total_price = product["price_per_day"] * product["days"]
availability_text = "Да" if product["is_available"] else "Нет"

print(f"Категория: {product['category']}")
print(f"Товар: {product['name']}")
print(f"Цена за день: {product['price_per_day']} грн")
print(f"Количество дней: {product['days']}")
print(f"Итого: {total_price} грн")
print(f"Залог: {product['deposit']} грн")
print(f"Доступен: {availability_text}")
