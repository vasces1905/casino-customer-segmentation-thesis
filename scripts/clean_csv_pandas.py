# clean date format problems and import to pstgre
# Optional: write validated data to new CSV
# df.to_csv("src/data/game_events_validated.csv", index=False)


import pandas as pd
from sqlalchemy import create_engine

csv_path = 'src/data/game_events_cleaned.csv'

# 1. CSV'yi oku
df = pd.read_csv(csv_path)

# 2. Hatalı tarihleri NaT olarak işaretle
df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
df['gameing_day'] = pd.to_datetime(df['gameing_day'], errors='coerce')

# 🎯 BONUS: Geçersiz tarihli satırları logla
invalid_rows = df[df['ts'].isna() | df['gameing_day'].isna()]
invalid_rows.to_csv("src/data/invalid_dates_log.csv", index=False)

# 3. Geçerli satırlarla devam et
df = df.dropna(subset=['ts', 'gameing_day'])

# 4. PostgreSQL'e aktar
engine = create_engine("postgresql://researcher:academic_password_2024@localhost:5432/casino_research")

df.to_sql(
    name='game_events',
    schema='casino_data',
    con=engine,
    if_exists='append',
    index=False,
    chunksize=10000,
    method='multi'
)
