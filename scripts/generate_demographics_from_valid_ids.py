import pandas as pd
import psycopg2
from faker import Faker
from datetime import datetime
from sqlalchemy import create_engine

faker = Faker()
engine = create_engine("postgresql://researcher:academic_password_2024@localhost:5432/casino_research")
conn = engine.raw_connection()
cursor = conn.cursor()

def get_valid_player_ids():
    query = "SELECT DISTINCT player_id FROM casino_data.temp_valid_game_events"
    return pd.read_sql(query, engine)["player_id"].tolist()

def generate_demographics(pid):
    age_bins = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    return {
        "customer_id": pid,
        "age_range": faker.random_element(elements=age_bins),
        "gender": faker.random_element(elements=["Male", "Female"]),
        "nationality": faker.country(),
        "registration_month": faker.date_between(start_date='-3y', end_date='today').strftime('%Y-%m')
    }

valid_ids = get_valid_player_ids()
print(f"Found {len(valid_ids)} valid player_ids.")

for pid in valid_ids:
    demo = generate_demographics(pid)
    cursor.execute("""
        INSERT INTO casino_data.customer_demographics (
            customer_id, age_range, gender, nationality, registration_month
        )
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (customer_id)
        DO UPDATE SET
            age_range = EXCLUDED.age_range,
            gender = EXCLUDED.gender,
            nationality = EXCLUDED.nationality,
            registration_month = EXCLUDED.registration_month;
    """, (
        demo["customer_id"],
        demo["age_range"],
        demo["gender"],
        demo["nationality"],
        demo["registration_month"]
    ))

conn.commit()
cursor.close()
conn.close()
print(" OK -  All demographics updated successfully.")
