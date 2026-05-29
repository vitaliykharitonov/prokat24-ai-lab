
products = [
    {
        "category": "Туристическое снаряжение",
        "name": "Палатка 4-местная",
        "price_per_day": 400,
        "deposit": 6000,
        "is_available": True,
    },
    {
        "category": "Туристическое снаряжение",
        "name": "Спальник",
        "price_per_day": 120,
        "deposit": 1000,
        "is_available": True,
    },
    {
        "category": "Туристическое снаряжение",
        "name": "Рюкзак туристический",
        "price_per_day": 150,
        "deposit": 1500,
        "is_available": False,
    },
    {
        "category": "Туристическое снаряжение",
        "name": "Термос",
        "price_per_day": 100,
        "deposit": 1000,
        "is_available": True,
    },
]

for product in products:
    availability_text = "Да" if product["is_available"] else "Нет"

    print(f"Категория: {product['category']}")
    print(f"Товар: {product['name']}")
    print(f"Цена за день: {product['price_per_day']} грн")
    print(f"Залог: {product['deposit']} грн")
    print(f"Доступен: {availability_text}")
    print()
