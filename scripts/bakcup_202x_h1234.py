import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://researcher:academic_password_2024@localhost:5432/casino_research")

query = "SELECT * FROM casino_data.kmeans_export_all_periods"
df = pd.read_sql(query, engine)

df.to_csv("C:/Users/ycanli/Documents/Source/casino-customer-segmentation-thesis/src/data/kmeans_export_all_periods.csv", index=False)
