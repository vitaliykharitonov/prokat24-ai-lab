product_name = "Палатка 4-местная"
price_per_day = 400
days = 5
deposit = 6000
is_available = True

total_price = price_per_day * days

availability_text = "Да" if is_available else "Нет"

print(f"Товар: {product_name}")
print(f"Цена за день: {price_per_day} грн")
print(f"Количество дней: {days}")
print(f"Итого: {total_price} грн")
print(f"Залог: {deposit} грн")
print(f"Доступен: {availability_text}")

print()
print("Типы данных:")
print(f"product_name: {type(product_name)}")
print(f"price_per_day: {type(price_per_day)}")
print(f"days: {type(days)}")
print(f"total_price: {type(total_price)}")
print(f"deposit: {type(deposit)}")
print(f"is_available: {type(is_available)}")
print(f"availability_text: {type(availability_text)}")
