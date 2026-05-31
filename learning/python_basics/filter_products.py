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

print("Проверяем товары до 120 грн/день:")

for product in products:
    if product["is_available"] and product["price_per_day"] <= 120:
        print(f"Подходит: {product['name']}: {product['price_per_day']} грн/день")
    else:
        print(f"Не подходит: {product['name']}")
