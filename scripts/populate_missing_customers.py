# scripts/populate_customers_from_game_events.py
# Generates matched demographic data based on player_ids from game_events table

import pandas as pd
import random
from faker import Faker
from sqlalchemy import create_engine, text
from datetime import datetime


# Initialize Faker and seed for reproducible data generation
fake = Faker()
random.seed(42)
Faker.seed(42)

# PostgreSQL connection
engine = create_engine("postgresql://researcher:academic_password_2024@localhost:5432/casino_research")

print("Connecting to database and fetching distinct player_ids...")

# Step 1: Get all valid player_ids from game_events
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT DISTINCT player_id 
        FROM casino_data.game_events 
        WHERE player_id IS NOT NULL;
    """))
    player_ids = [row[0] for row in result]

print(f"Total unique player_ids found: {len(player_ids)}")

# Step 2: Define demographic data generator
def generate_fake_customer(player_id):
    age_ranges = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    genders = ["Male", "Female"]
    regions = [
        "Germany", "Turkey", "Bulgaria", "Greece", "Kosovo",
        "Netherlands", "Azerbaijan", "Poland", "Romania", "Italy"
    ]
    #registration_date = fake.date_between(start_date='2022-01-01', end_date='2022-12-31')
    registration_date = fake.date_between(start_date=datetime(2022, 1, 1), end_date=datetime(2022, 12, 31)
)

    return {
        "customer_id": player_id,
        "age_range": random.choice(age_ranges),
        "gender": random.choice(genders),
        "region": random.choice(regions),
        "registration_month": registration_date.strftime("%Y-%m"),
        "customer_segment": None,
        "import_session_id": None,
        "created_at": pd.Timestamp.now()
    }

print("Generating demographic data...")

# Step 3: Create demographic records
customers_data = [generate_fake_customer(pid) for pid in player_ids]
df_customers = pd.DataFrame(customers_data)

print(f"Total rows generated: {len(df_customers)}")

# Step 4: Clear existing data
print("Clearing existing data in casino_data.customer_demographics...")
with engine.begin() as conn:
    #conn.execute(text("TRUNCATE TABLE casino_data.customer_demographics;"))
    conn.execute(text("DELETE FROM casino_data.customer_demographics;"))

# Step 5: Insert new data
print("Inserting new demographic data into database...")
df_customers.to_sql(
    name="customer_demographics",
    schema="casino_data",
    con=engine,
    if_exists="append",
    index=False,
    chunksize=10000,
    method="multi"
)

print("Process completed: All demographic data written based on game_events.player_id.")
