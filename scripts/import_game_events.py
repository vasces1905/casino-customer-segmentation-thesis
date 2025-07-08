import pandas as pd
from sqlalchemy import create_engine

# CSV yolu ve chunk
csv_path = "src/data/game_events_cleaned.csv"
chunk_size = 10000

# PostgreSQL bağlantısı (şifre, port vb. sana göre ayarlanmalı)
engine = create_engine("postgresql+psycopg2://postgres:<şifre>@localhost:5432/casino_research")

# Tarih alanları için özel dönüştürücü
date_columns = ["ts", "gameing_day"]

# Chunk işle
for chunk in pd.read_csv(
    csv_path,
    chunksize=chunk_size,
    parse_dates=date_columns,
    na_values=["0000-00-00", "0000-00-00 00:00:00"]
):
    # Tarih formatı geçersiz olanları NaT olarak set ediyoruz
    for col in date_columns:
        chunk[col] = pd.to_datetime(chunk[col], errors='coerce')

    chunk.to_sql(
        name='game_events',
        con=engine,
        schema='casino_data',
        if_exists='append',
        index=False,
        method='multi'
    )

print("OK - Data has loaded successfully.")
