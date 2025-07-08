import pandas as pd
from config.db_config import get_db_connection, MAIN_SCHEMA
import os

def load_demographics_csv_to_db(csv_path: str):
    """
    Loads customer_demographics_2022.csv into PostgreSQL target table.
    Assumes schema and table already exist.
    """
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Make sure customer_id is string
    df['customer_id'] = df['customer_id'].astype(str)

    # Build records manually as list of tuples (prevents int64 conversion)
    records = list(df[['customer_id', 'age_range', 'gender', 'region', 'registration_month', 'customer_segment']].itertuples(index=False, name=None))

    with get_db_connection() as conn:
        cursor = conn.cursor()

        insert_query = f"""
        INSERT INTO {MAIN_SCHEMA}.customer_demographics
        (customer_id, age_range, gender, region, registration_month, customer_segment)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (customer_id) DO NOTHING;
        """

        cursor.executemany(insert_query, records)
        conn.commit()
        print(f"Inserted {cursor.rowcount} new rows into customer_demographics")
