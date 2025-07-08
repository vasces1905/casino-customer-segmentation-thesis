# scripts/check_record_count.py

import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

conn = psycopg2.connect(
    host=os.getenv("DB_HOST", "localhost"),
    port=os.getenv("DB_PORT", "5432"),
    dbname=os.getenv("DB_NAME", "casino_research"),
    user=os.getenv("DB_USER", "researcher"),
    password=os.getenv("DB_PASSWORD", "academic_password_2024")
)

cur = conn.cursor()
cur.execute("SELECT COUNT(*) FROM casino_data.customer_demographics")
count = cur.fetchone()[0]
print(f"✅ Total customer records in DB: {count}")

cur.close()
conn.close()
