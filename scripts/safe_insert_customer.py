from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine("postgresql://researcher:academic_password_2024@localhost:5432/casino_research")

def safe_insert(df: pd.DataFrame, table: str):
    """
    INSERT öncesi valid_customer_ids view'e göre müşteri ID'lerini filtreler.
    """
    with engine.connect() as conn:
        valid_ids = pd.read_sql("SELECT customer_id FROM casino_data.valid_customer_ids", conn)
        df_filtered = df[df['customer_id'].isin(valid_ids['customer_id'])]
        
        if df_filtered.empty:
            print("Uygun müşteri bulunamadı, işlem yapılmadı.")
            return

        df_filtered.to_sql(table, con=conn, schema='casino_data', if_exists='append', index=False)
        print(f"✅ {len(df_filtered)} satır başarıyla eklendi → {table}")

# Örnek kullanım
# df = pd.read_csv("insert_data.csv")
# safe_insert(df, "customer_temporal_features")
