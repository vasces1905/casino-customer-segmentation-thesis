# debug_database.py
"""
Database Debug Script for Academic Research
University of Bath - MSc Computer Science
Student: Muhammed Yavuzhan CANLI
Ethics Approval: 10351-12382

Debug PostgreSQL database connection and content
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



import psycopg2
import pandas as pd
from src.config.db_config import get_db_config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_database_connection():
    """Debug database connection and check table structure."""
    
    logger.info("=== ACADEMIC DATABASE DEBUG SESSION ===")
    logger.info("University of Bath - Ethics Approval: 10351-12382")
    
    try:
        # Get database configuration
        db_config = get_db_config()
        logger.info(f"Connecting to database: {db_config.get('host', 'localhost')}:{db_config.get('port', 5432)}")
        
        # Connect to database
        conn = psycopg2.connect(**db_config)
        cursor = conn.cursor()
        
        logger.info("✅ Database connection successful")
        
        # Check if schemas exist
        logger.info("\n1. CHECKING SCHEMAS:")
        cursor.execute("""
            SELECT schema_name 
            FROM information_schema.schemata 
            WHERE schema_name IN ('casino_data', 'academic_audit')
            ORDER BY schema_name
        """)
        schemas = cursor.fetchall()
        
        for schema in schemas:
            logger.info(f"✅ Schema exists: {schema[0]}")
        
        if not schemas:
            logger.error("❌ No required schemas found!")
            return
            
        # Check tables in casino_data schema
        logger.info("\n2. CHECKING TABLES IN casino_data:")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'casino_data'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        
        for table in tables:
            logger.info(f"✅ Table exists: casino_data.{table[0]}")
            
        # Check customer_demographics table structure
        if ('customer_demographics',) in tables:
            logger.info("\n3. CUSTOMER_DEMOGRAPHICS TABLE STRUCTURE:")
            cursor.execute("""
                SELECT column_name, data_type, character_maximum_length, is_nullable
                FROM information_schema.columns 
                WHERE table_schema = 'casino_data' 
                  AND table_name = 'customer_demographics'
                ORDER BY ordinal_position
            """)
            columns = cursor.fetchall()
            
            for col in columns:
                logger.info(f"  • {col[0]}: {col[1]}" + 
                          (f"({col[2]})" if col[2] else "") + 
                          (f" NULL" if col[3] == 'YES' else " NOT NULL"))
            
            # Check if customer_id is still INTEGER
            customer_id_type = next((col[1] for col in columns if col[0] == 'customer_id'), None)
            if customer_id_type == 'integer':
                logger.error("❌ PROBLEM: customer_id is still INTEGER type!")
                logger.error("   This will cause 'integer out of range' error")
                logger.info("   ➤ Need to run schema fix: ALTER TABLE ... ALTER COLUMN customer_id TYPE VARCHAR(50)")
            elif customer_id_type:
                logger.info(f"✅ customer_id type: {customer_id_type}")
            
            # Check table content
            logger.info("\n4. TABLE CONTENT CHECK:")
            cursor.execute("SELECT COUNT(*) FROM casino_data.customer_demographics")
            count = cursor.fetchone()[0]
            logger.info(f"Records in customer_demographics: {count}")
            
            if count > 0:
                cursor.execute("""
                    SELECT customer_id, age_range, gender, region 
                    FROM casino_data.customer_demographics 
                    LIMIT 5
                """)
                sample_data = cursor.fetchall()
                logger.info("Sample records:")
                for record in sample_data:
                    logger.info(f"  • {record}")
            else:
                logger.info("❌ Table is empty - this explains why pipeline uses synthetic data")
        
        # Check academic audit tables
        logger.info("\n5. CHECKING ACADEMIC AUDIT TABLES:")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'academic_audit'
            ORDER BY table_name
        """)
        audit_tables = cursor.fetchall()
        
        for table in audit_tables:
            logger.info(f"✅ Audit table: academic_audit.{table[0]}")
            
        # Check recent import attempts
        if ('access_log',) in audit_tables:
            cursor.execute("""
                SELECT action, record_count, timestamp, academic_purpose
                FROM academic_audit.access_log 
                ORDER BY timestamp DESC 
                LIMIT 5
            """)
            recent_logs = cursor.fetchall()
            
            if recent_logs:
                logger.info("\n6. RECENT IMPORT ATTEMPTS:")
                for log in recent_logs:
                    logger.info(f"  • {log[2]}: {log[0]} ({log[1]} records) - {log[3]}")
            else:
                logger.info("\n6. No import attempts logged")
        
        cursor.close()
        conn.close()
        
        logger.info("\n=== DEBUG SUMMARY ===")
        logger.info("✅ Database connection working")
        logger.info("✅ Schema and tables exist")
        
        if customer_id_type == 'integer':
            logger.error("❌ MAIN ISSUE: customer_id is INTEGER type")
            logger.info("SOLUTION: Run schema fix script to change to VARCHAR(50)")
        elif count == 0:
            logger.error("❌ MAIN ISSUE: customer_demographics table is empty")
            logger.info("SOLUTION: Fix customer_id type first, then retry CSV import")
        else:
            logger.info("✅ Database appears healthy")
            
    except Exception as e:
        logger.error(f"❌ Database debug failed: {e}")
        logger.info("Check database connection settings and Docker container status")

def check_csv_file():
    """Check if CSV file exists and analyze its content."""
    
    logger.info("\n=== CSV FILE CHECK ===")
    
    csv_paths = [
        "src/data/customer_demographics_2022.csv",
        "customer_demographics_2022.csv",
        "data/customer_demographics_2022.csv"
    ]
    
    csv_found = False
    for path in csv_paths:
        if os.path.exists(path):
            logger.info(f"✅ Found CSV file: {path}")
            csv_found = True
            
            # Analyze CSV content
            try:
                df = pd.read_csv(path)
                logger.info(f"CSV records: {len(df)}")
                logger.info(f"CSV columns: {list(df.columns)}")
                
                # Check problematic customer_ids
                if 'customer_id' in df.columns:
                    sample_ids = df['customer_id'].head(3).tolist()
                    logger.info(f"Sample customer_ids: {sample_ids}")
                    
                    # Check for oversized IDs
                    def extract_numeric(cid):
                        try:
                            return int(str(cid).replace('CUST_', ''))
                        except:
                            return 0
                    
                    df['numeric_id'] = df['customer_id'].apply(extract_numeric)
                    max_id = df['numeric_id'].max()
                    
                    if max_id > 2147483647:
                        logger.error(f"❌ CSV contains oversized customer_id: {max_id:,}")
                        logger.error("   This will cause PostgreSQL INTEGER overflow")
                        logger.info("   ➤ Need to fix customer_id format before import")
                    else:
                        logger.info(f"✅ Customer_id values are within range: max {max_id:,}")
                        
            except Exception as e:
                logger.error(f"❌ Could not analyze CSV: {e}")
            break
    
    if not csv_found:
        logger.error("❌ CSV file not found in any expected location")
        logger.info("Expected locations:")
        for path in csv_paths:
            logger.info(f"  • {path}")

def main():
    """Run complete database debug session."""
    
    print("CASINO CUSTOMER SEGMENTATION - DATABASE DEBUG")
    print("=" * 50)
    print("University of Bath - MSc Computer Science")
    print("Student: Muhammed Yavuzhan CANLI")
    print("Ethics Approval: 10351-12382")
    print("=" * 50)
    
    # Check CSV file first
    check_csv_file()
    
    # Check database
    debug_database_connection()
    
    print("\n" + "=" * 50)
    print("DEBUG SESSION COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    main()