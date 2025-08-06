#!/usr/bin/env python3
"""
db_connection_test.py

PostgreSQL bağlantı test script'i - Doğru credentials ile
"""

import psycopg2
import pandas as pd
from sqlalchemy import create_engine, text

# Doğru bağlantı bilgileri
DB_CONFIG = {
    'host': 'localhost',
    'database': 'casino_research',
    'user': 'researcher',
    'password': 'academic_password_2024'
}

def test_connection():
    """Bağlantı test et ve tablo yapısını incele"""
    try:
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        )
        
        print("🔗 Database connection successful!")
        
        # 1. casino_data şemasındaki tabloları listele
        tables_query = """
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'casino_data'
        ORDER BY table_name
        """
        tables_df = pd.read_sql(tables_query, engine)
        print(f"📋 Tables in casino_data: {tables_df['table_name'].tolist()}")
        
        # 2. customer_features tablosu kolonları
        if 'customer_features' in tables_df['table_name'].values:
            columns_query = """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'casino_data' 
            AND table_name = 'customer_features'
            ORDER BY ordinal_position
            """
            columns_df = pd.read_sql(columns_query, engine)
            print(f"\n📊 customer_features columns:")
            for _, row in columns_df.iterrows():
                print(f"   {row['column_name']} ({row['data_type']})")
            
            # Sample data
            sample_query = "SELECT * FROM casino_data.customer_features LIMIT 3"
            sample_df = pd.read_sql(sample_query, engine)
            print(f"\n🔍 Sample data (3 rows):")
            print(sample_df.head())
        
        # 3. promo_label tablosu
        if 'promo_label' in tables_df['table_name'].values:
            promo_columns_query = """
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_schema = 'casino_data' 
            AND table_name = 'promo_label'
            ORDER BY ordinal_position
            """
            promo_columns_df = pd.read_sql(promo_columns_query, engine)
            print(f"\n📊 promo_label columns:")
            for _, row in promo_columns_df.iterrows():
                print(f"   {row['column_name']} ({row['data_type']})")
            
            # Period check
            periods_query = "SELECT DISTINCT period FROM casino_data.promo_label ORDER BY period"
            periods_df = pd.read_sql(periods_query, engine)
            print(f"\n📅 Available periods: {periods_df['period'].tolist()}")
            
            # Count per period
            count_query = """
            SELECT period, COUNT(*) as count 
            FROM casino_data.promo_label 
            GROUP BY period 
            ORDER BY period
            """
            count_df = pd.read_sql(count_query, engine)
            print(f"\n📈 Records per period:")
            print(count_df.to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

if __name__ == "__main__":
    test_connection()