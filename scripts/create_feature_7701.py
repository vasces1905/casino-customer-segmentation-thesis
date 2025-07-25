import pandas as pd
from sqlalchemy import create_engine

# PostgreSQL bağlantısı
engine = create_engine("postgresql://researcher:academic_password_2024@localhost:5432/casino_research")

# Eksik müşteri verileri çekiliyor
query = """
SELECT
    player_id AS customer_id,
    bet,
    win,
    ts AS timestamp,
    gameing_day
FROM casino_data.temp_valid_game_events
WHERE player_id IN (
    SELECT customer_id FROM casino_data.customers_missing_features
);
"""

df_missing_events = pd.read_sql(query, engine)
