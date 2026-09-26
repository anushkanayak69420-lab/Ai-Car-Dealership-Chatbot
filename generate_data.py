"""
Generates a realistic synthetic car dealership dataset: inventory + sales.
Mirrors the kind of data a small car dealership (like the one from your
internship) would track — stock in, stock sold, days on lot, price, etc.

Run this once to create data/car_sales.csv
"""

import pandas as pd
import numpy as np
import os

np.random.seed(7)

brands_models = {
    "Maruti Suzuki": ["Swift", "Baleno", "WagonR", "Ertiga"],
    "Hyundai": ["Creta", "i20", "Venue", "Verna"],
    "Tata": ["Nexon", "Punch", "Harrier", "Altroz"],
    "Honda": ["City", "Amaze"],
    "Toyota": ["Innova", "Fortuner"],
    "Kia": ["Seltos", "Sonet"],
}

body_types = {
    "Swift": "Hatchback", "Baleno": "Hatchback", "WagonR": "Hatchback", "Ertiga": "MUV",
    "Creta": "SUV", "i20": "Hatchback", "Venue": "SUV", "Verna": "Sedan",
    "Nexon": "SUV", "Punch": "SUV", "Harrier": "SUV", "Altroz": "Hatchback",
    "City": "Sedan", "Amaze": "Sedan", "Innova": "MUV", "Fortuner": "SUV",
    "Seltos": "SUV", "Sonet": "SUV",
}

base_price = {
    "Swift": 650000, "Baleno": 700000, "WagonR": 550000, "Ertiga": 900000,
    "Creta": 1200000, "i20": 750000, "Venue": 950000, "Verna": 1150000,
    "Nexon": 950000, "Punch": 700000, "Harrier": 1650000, "Altroz": 700000,
    "City": 1250000, "Amaze": 800000, "Innova": 2000000, "Fortuner": 3500000,
    "Seltos": 1150000, "Sonet": 850000,
}

branches = ["Downtown Showroom", "Highway Branch", "City Mall Outlet"]

# Seasonal demand boost by month (festive season / new year / financial year-end
# are realistic peaks for Indian car sales)
seasonal_boost = {
    "Creta": {10: 1.8, 11: 2.0, 3: 1.5},
    "Nexon": {10: 1.7, 11: 1.9, 3: 1.6},
    "Fortuner": {10: 1.6, 11: 1.9},
    "Innova": {5: 1.5, 6: 1.4},   # wedding season
    "Swift": {1: 1.6, 3: 1.7},    # new year + year-end discounts
    "WagonR": {1: 1.5, 3: 1.6},
    "Harrier": {10: 1.9, 11: 2.1},
    "Seltos": {10: 1.7, 11: 1.8},
}

rows = []
sale_id = 5000

for month in range(1, 13):
    num_sales = np.random.randint(35, 55)
    for _ in range(num_sales):
        brand = np.random.choice(list(brands_models.keys()))
        model = np.random.choice(brands_models[brand])
        body_type = body_types[model]
        branch = np.random.choice(branches)
        day = np.random.randint(1, 28)
        date = pd.Timestamp(year=2024, month=month, day=day)

        multiplier = seasonal_boost.get(model, {}).get(month, 1.0)
        # simulate whether this listing actually sold this month (weighted by seasonality)
        if np.random.random() > (0.55 / multiplier):
            continue  # didn't sell, skip (keeps seasonal peaks realistic)

        price = round(base_price[model] * np.random.uniform(0.95, 1.08), -3)
        days_in_inventory = max(3, int(np.random.normal(28, 12) / multiplier))
        color = np.random.choice(["White", "Silver", "Red", "Black", "Blue", "Grey"])

        rows.append({
            "sale_id": sale_id,
            "date": date.strftime("%Y-%m-%d"),
            "month": date.strftime("%B"),
            "month_num": month,
            "brand": brand,
            "model": model,
            "body_type": body_type,
            "branch": branch,
            "color": color,
            "price": price,
            "days_in_inventory": days_in_inventory,
        })
        sale_id += 1

df = pd.DataFrame(rows)
os.makedirs("data", exist_ok=True)
df.to_csv("data/car_sales.csv", index=False)

print(f"Generated {len(df)} car sale records -> data/car_sales.csv")
print(df.head())
print("\nTotal sales by model:")
print(df.groupby("model").size().sort_values(ascending=False))
