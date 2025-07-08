import psycopg2

conn = psycopg2.connect(
    dbname='casino_research',
    user='researcher',
    password='academic_password_2024',
    host='localhost',
    port=5432
)

cur = conn.cursor()

with open("src/data/game_events_cleaned.csv", 'r', encoding='utf-8') as f:
    next(f)  # skip header
    cur.copy_expert(
        """
        COPY casino_data.game_events (
            id, ts, player_id, event_id, locno, win, bet, credit,
            point, denom, promo_bet, gameing_day, asset, curr_type, avg_bet
        )
        FROM STDIN WITH CSV
        """,
        f
    )

conn.commit()
cur.close()
conn.close()

print("The Data has Loaded Successfully.")
