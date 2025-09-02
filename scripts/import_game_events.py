import pandas as pd
from sqlalchemy import create_engine

# CSV path and chunk size
csv_path = "src/data/game_events_cleaned.csv"
chunk_size = 10000

# PostgreSQL connection (password, port etc. should be configured accordingly)
engine = create_engine("postgresql+psycopg2://postgres:<password>@localhost:5432/casino_research")

# Special converter for date fields
date_columns = ["ts", "gameing_day"]

# Process chunks
for chunk in pd.read_csv(
    csv_path,
    chunksize=chunk_size,
    parse_dates=date_columns,
    na_values=["0000-00-00", "0000-00-00 00:00:00"]
):
    # Set invalid date formats to NaT (Not a Time)
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

print("Game events data loading completed.")
