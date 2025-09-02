# Player Sessions Data Loader
# University of Bath - MSc Computer Science
# Student: Muhammed Yavuzhan Canli


import pandas as pd
import psycopg2
from src.config.db_config import get_db_config

# Load player sessions data from CSV
df = pd.read_csv('data/player_sessions_2022.csv')

# Insert data into PostgreSQL database
conn = psycopg2.connect(**get_db_config())
cursor = conn.cursor()

for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO casino_data.player_sessions (
            customer_id, session_start, session_end,
            total_bet, total_win, game_type, machine_id
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, tuple(row))

conn.commit()
conn.close()
