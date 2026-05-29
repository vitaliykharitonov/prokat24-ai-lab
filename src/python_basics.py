
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