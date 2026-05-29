import json
from os import name
from pathlib import Path


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
]

project_root = Path(__file__).resolve().parent.parent
output_file = project_root / "data" / "products_sample.json"

with open(output_file, "w", encoding="utf-8") as file:
    json.dump(products, file, ensure_ascii=False, indent=2)

print(f"Saved {len(products)} products to {output_file}")

with open(output_file, "r", encoding="utf-8") as file:
    loaded_products = json.load(file)

print(f"Loaded {len(loaded_products)} products from {output_file}")
print(f"First product: {loaded_products[0]['name']}")
