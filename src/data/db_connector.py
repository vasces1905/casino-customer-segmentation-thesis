# src/data/db_connector.py

"""
Academic-Compliant Database Connector
====================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution
"""

""" 
Technical Explanation & What are included
- Connects to PostgreSQL (with psycopg2)
- It logs every connection academically: writes to access_log table
- There is a test_connection() function to see if it can connect to the database
   Note AUDIT_SCHEMA.access_log table needs to be defined under schema/
   
*** AcademicDBConnector
- get_connection(): Opens the connection → performs the operation → closes and logs
- _log_access(): Writes a record to the academic_audit.access_log table as CONNECTION_ESTABLISHED
- test_connection(): Tests the connection (SELECT 1)
   Note: The AUDIT_SCHEMA.access_log table must be defined under schema/.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any 
import psycopg2  # Connects to PostgreSQL
from psycopg2.extras import RealDictCursor
from contextlib import contextmanager

from ..config.db_config import DB_CONFIG, AUDIT_SCHEMA
from ..config.academic_config import ACADEMIC_METADATA

logger = logging.getLogger(__name__)


class AcademicDBConnector:
    """
    Database connector with academic compliance and audit trail.
    Ensures all data access is logged for ethics compliance.
    """
    
    def __init__(self):
        self.ethics_ref = ACADEMIC_METADATA["ethics_ref"]
        self.student_id = ACADEMIC_METADATA["student_id"]
        self.connection_params = DB_CONFIG.copy()
        
    @contextmanager
    def get_connection(self):
        """Context manager for database connections with audit logging"""
        conn = None
        try:
            conn = psycopg2.connect(**self.connection_params)
            
            # Log connection for academic audit
            self._log_access(conn, "CONNECTION_ESTABLISHED")
            
            yield conn
            
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                self._log_access(conn, "CONNECTION_CLOSED")
                conn.close()
    
    def _log_access(self, conn, action: str):
        """Log all database access for academic compliance"""
        with conn.cursor() as cur:
            cur.execute(f"""
                INSERT INTO {AUDIT_SCHEMA}.access_log 
                (student_id, ethics_ref, action, timestamp)
                VALUES (%s, %s, %s, %s)
            """, (self.student_id, self.ethics_ref, action, datetime.now()))
            conn.commit()
    
    def test_connection(self) -> bool:
        """Test database connectivity"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT 1")
                    return cur.fetchone()[0] == 1
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False