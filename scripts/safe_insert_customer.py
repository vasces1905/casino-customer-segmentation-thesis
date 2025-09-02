from sqlalchemy import create_engine, text
import pandas as pd

engine = create_engine("postgresql://researcher:academic_password_2024@localhost:5432/casino_research")

def safe_insert(df: pd.DataFrame, table: str):
    """
    Filters customer IDs against valid_customer_ids view before INSERT operation.
    """
    with engine.connect() as conn:
        valid_ids = pd.read_sql("SELECT customer_id FROM casino_data.valid_customer_ids", conn)
        df_filtered = df[df['customer_id'].isin(valid_ids['customer_id'])]
        
        if df_filtered.empty:
            print("No matching customers found, operation skipped.")
            return

        df_filtered.to_sql(table, con=conn, schema='casino_data', if_exists='append', index=False)
        print(f"{len(df_filtered)} records successfully inserted into {table}")

# Example usage:
# df = pd.read_csv("insert_data.csv")
# safe_insert(df, "customer_temporal_features")
