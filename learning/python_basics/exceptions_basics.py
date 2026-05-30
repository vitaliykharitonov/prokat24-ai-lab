
product = {
    "name": "Палатка 4-местная",
    "price_per_day": 400,
}

deposit = product.get("deposit", "Залог не указан")
print(product["name"])
print(deposit)
