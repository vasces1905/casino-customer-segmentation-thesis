# game sessions couldn't take to DB. 
# Will be provided by this script  - 03.07.2025
# Muhammed Yavuzhan Canli


import pandas as pd
import psycopg2
from src.config.db_config import get_db_config

# Read from csv
df = pd.read_csv('data/player_sessions_2022.csv')

# Write to DB
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
