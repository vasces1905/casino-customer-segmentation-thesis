# scripts/import_casino_batch_data.py

"""
Import Pre-Anonymized Casino Batch Data
=======================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

IMPORTANT NOTE:
--------------
This script imports PRE-ANONYMIZED data provided by Casino IT Department.
We do NOT connect directly to casino production database.
All data has been anonymized by Casino IT before delivery.

Data Flow:
----------
1. Casino IT exports and anonymizes data from production MSSQL
2. Data delivered as CSV/Excel files via secure transfer (RDP/Secure FTP)
3. This script imports to our PostgreSQL research database
4. Additional privacy measures applied (temporal noise, zone generalization)

Academic Integrity:
------------------
This approach ensures compliance with:
- Ethics approval 10351-12382 
- GDPR Article 4(5)
- Casino internal security policies
- University data handling requirements
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from typing import Dict, List, Optional
import hashlib

from src.config.academic_config import ACADEMIC_METADATA
from src.data.db_connector import AcademicDBConnector
from src.data.anonymizer import AcademicDataAnonymizer

# Configure logging with timestamps just in case there is any check for 
# anonymised data while getting it from a real casino environment.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'import_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CasinoBatchImporter:
    """
    Imports pre-anonymized casino data from batch files.
    
    Academic Note:
        We receive data that has already been anonymized by Casino IT.
        Customer IDs are already in CUST_XXXXXX format.
        We add additional privacy layers as per ethics requirements.
    """
    
    def __init__(self, batch_dir: str = 'data/casino_batch'):
        """
        Initialize importer for pre-anonymized batch files.
        
        Args:
            batch_dir: Directory containing batch files from casino
        """
        self.batch_dir = batch_dir
        self.db_connector = AcademicDBConnector()
        self.anonymizer = AcademicDataAnonymizer()
        self.import_timestamp = datetime.now()
        
        # Create import session ID for tracking
        self.import_session_id = f"IMP_{self.import_timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        logger.info("="*60)
        logger.info(f"BATCH IMPORT SESSION: {self.import_session_id}")
        logger.info(f"Timestamp: {self.import_timestamp}")
        logger.info(f"Reading from: {self.batch_dir}")
        logger.info(f"Ethics approval: {ACADEMIC_METADATA['ethics_ref']}")
        logger.info("="*60)
    
    def log_import_metadata(self, filename: str, record_count: int, 
                          table_name: str, additional_info: Dict = None):
        """
        Log detailed import metadata with timestamps.
        
        Args:
            filename: Source file name
            record_count: Number of records imported
            table_name: Target table name
            additional_info: Any additional metadata
        """
        log_entry = {
            "import_session_id": self.import_session_id,
            "timestamp": datetime.now().isoformat(),
            "filename": filename,
            "record_count": record_count,
            "target_table": table_name,
            "ethics_ref": ACADEMIC_METADATA['ethics_ref'],
            "additional_info": additional_info or {}
        }
        
        logger.info(f"IMPORT LOG: {log_entry}")
        
        # Also save to database audit trail
        try:
            with self.db_connector.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO academic_audit.data_provenance 
                        (original_source, target_table, records_affected, 
                         anonymization_method, transformation_applied,
                         academic_purpose, created_at, created_by)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        filename,
                        f'casino_data.{table_name}',
                        record_count,
                        'Pre-anonymized by Casino IT',
                        'Additional temporal noise and zone generalization',
                        f'MSc thesis research - Import session {self.import_session_id}',
                        datetime.now(),
                        ACADEMIC_METADATA['student_id']
                    ))
                conn.commit()
        except Exception as e:
            logger.error(f"Failed to log to audit table: {e}")
    
    def verify_file_integrity(self, file_path: str) -> Optional[str]:
        """
        Verify file integrity and log file metadata.
        
        Returns:
            File hash for verification
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Calculate file hash
        file_hash = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                file_hash.update(chunk)
        
        file_info = {
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024),
            "file_modified": datetime.fromtimestamp(os.path.getmtime(file_path)),
            "file_hash": file_hash.hexdigest()
        }
        
        logger.info(f"File verification: {file_info}")
        return file_hash.hexdigest()
    
    def import_customer_demographics(self):
        """
        Import pre-anonymized customer demographics.
        
        Expected file: customer_demographics_batch_YYYYMMDD.csv
        Expected columns (from Casino IT):
        - CustomerID (already CUST_XXXXXX format)
        - AgeRange (already generalized: 18-24, 25-34, etc.)
        - Gender (M/F)
        - Region (already generalized)
        - RegistrationMonth (YYYY-MM)
        """
        start_time = datetime.now()
        logger.info(f"\n[{start_time}] Starting customer demographics import...")
        
        # Find latest batch file
        files = [f for f in os.listdir(self.batch_dir) 
                if f.startswith('customer_demographics_batch_')]
        
        if not files:
            raise FileNotFoundError("No customer demographics batch file found")
        
        latest_file = sorted(files)[-1]
        file_path = os.path.join(self.batch_dir, latest_file)
        
        logger.info(f"Reading: {latest_file}")
        
        # Verify file integrity
        file_hash = self.verify_file_integrity(file_path)
        
        # Read pre-anonymized data
        df = pd.read_csv(file_path)
        initial_count = len(df)
        
        logger.info(f"Initial record count: {initial_count}")
        
        # Verify expected columns
        expected_cols = ['CustomerID', 'AgeRange', 'Gender', 'Region', 'RegistrationMonth']
        missing_cols = [col for col in expected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing expected columns: {missing_cols}")
        
        # Verify anonymization
        if not df['CustomerID'].str.match(r'^CUST_[A-Z0-9]{6}$').all():
            raise ValueError("Customer IDs not properly anonymized!")
        
        # Rename to match our schema
        df.rename(columns={
            'CustomerID': 'customer_id',
            'AgeRange': 'age_range',
            'Gender': 'gender',
            'Region': 'region',
            'RegistrationMonth': 'registration_month'
        }, inplace=True)
        
        # Add import metadata
        df['import_session_id'] = self.import_session_id
        df['import_timestamp'] = datetime.now()
        
        # Log completion
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        import_info = {
            "age_distribution": df['age_range'].value_counts().to_dict(),
            "gender_distribution": df['gender'].value_counts().to_dict(),
            "region_distribution": df['region'].value_counts().to_dict(),
            "duration_seconds": duration,
            "file_hash": file_hash
        }
        
        self.log_import_metadata(
            filename=latest_file,
            record_count=len(df),
            table_name='customer_demographics',
            additional_info=import_info
        )
        
        logger.info(f"[{end_time}] Completed in {duration:.2f} seconds")
        
        return df
    
    def import_player_sessions(self):
        """
        Import pre-anonymized player sessions with additional temporal noise.
        
        Expected file: player_sessions_batch_YYYYMMDD.csv
        
        Additional anonymization applied:
        - Add ±30 minute temporal noise to timestamps
        - Generalize machine zones
        """
        start_time = datetime.now()
        logger.info(f"\n[{start_time}] Starting player sessions import...")
        
        # Find latest batch file
        files = [f for f in os.listdir(self.batch_dir) 
                if f.startswith('player_sessions_batch_')]
        
        if not files:
            raise FileNotFoundError("No player sessions batch file found")