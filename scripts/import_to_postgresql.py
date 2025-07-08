from sqlalchemy import create_engine

engine = create_engine("postgresql://postgres:<şifreniz>@localhost:5432/casino_research")

df.to_sql(
    name='game_events',
    schema='casino_data',
    con=engine,
    if_exists='append',
    index=False,
    chunksize=10000,
    method='multi'
)
