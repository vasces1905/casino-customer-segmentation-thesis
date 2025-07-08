# test_db_connection.py
"""
PostgreSQL Connection Test for BATH University Thesis
Student: Muhammed Yavuzhan CANLI
Ethics Approval: 10351-12382
"""

import os
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_postgresql_connection():
    """Test PostgreSQL Docker connection"""
    try:
        # Connection parameters from .env
        conn_params = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': int(os.getenv('DB_PORT', 5432)),
            'database': os.getenv('DB_NAME', 'casino_research'),
            'user': os.getenv('DB_USER', 'researcher'),
            'password': os.getenv('DB_PASSWORD', 'academic_password_2024')
        }
        
        print("Testing PostgreSQL connection...")
        print(f"Host: {conn_params['host']}:{conn_params['port']}")
        print(f"Database: {conn_params['database']}")
        print(f"User: {conn_params['user']}")
        
        # Connect to PostgreSQL
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()
        print(f" OK- PostgreSQL version: {version[0]}")
        
        # Test academic audit tables
        cursor.execute("SELECT COUNT(*) FROM academic_audit.access_log;")
        audit_count = cursor.fetchone()[0]
        print(f" OK- Academic audit records: {audit_count}")
        
        # Test casino data tables
        cursor.execute("SELECT COUNT(*) FROM casino_data.customer_demographics;")
        customer_count = cursor.fetchone()[0]
        print(f" OK- Customer demographics records: {customer_count}")
        
        # Test table list
        cursor.execute("""
            SELECT table_schema, table_name 
            FROM information_schema.tables 
            WHERE table_schema IN ('academic_audit', 'casino_data')
            ORDER BY table_schema, table_name;
        """)
        tables = cursor.fetchall()
        
        print(f"\n Available tables ({len(tables)}):")
        for schema, table in tables:
            print(f"  - {schema}.{table}")
        
        # Academic compliance check
        cursor.execute("""
            INSERT INTO academic_audit.access_log 
            (student_id, ethics_ref, action, table_accessed, query_type, record_count)
            VALUES ('mycc21', '10351-12382', 'CONNECTION_TEST', 'multiple', 'SELECT', %s)
        """, (len(tables),))
        
        conn.commit()
        print(f"\n OK- Academic compliance logged")
        
        cursor.close()
        conn.close()
        
        print(f"\n OK- PostgreSQL connection test SUCCESSFUL!")
        return True
        
    except Exception as e:
        print(f" XXX- Connection test FAILED: {e}")
        return False

if __name__ == "__main__":
    test_postgresql_connection()