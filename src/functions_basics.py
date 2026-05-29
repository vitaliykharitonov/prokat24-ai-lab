
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
        "price_per_day": 120,
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

def is_suitable_product(product):
    return product["is_available"] and product["price_per_day"] <= 150

print("Подходящие товары:")

for product in products:
    if is_suitable_product(product):
        print(f"- {product['name']}: {product['price_per_day']} грн/день")