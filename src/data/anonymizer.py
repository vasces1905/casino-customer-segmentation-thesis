# src/data/anonymizer.py

"""
GDPR-Compliant Data Anonymization Module
========================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution
"""

import hashlib
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime, timedelta
import numpy as np

from ..config.academic_config import ACADEMIC_METADATA


class AcademicDataAnonymizer:
    """
    Implements GDPR Article 4(5) compliant anonymization.
    Ensures data cannot be re-identified while maintaining analytical value.
    """
    
    def __init__(self):
        self.ethics_ref = ACADEMIC_METADATA["ethics_ref"]
        self.salt = f"BATH_{self.ethics_ref}_ACADEMIC"  # Fixed salt for reproducibility
        
    def anonymize_customer_id(self, customer_id: str) -> str:
        """
        Convert customer ID to irreversible hash.
        Maintains referential integrity across tables.
        """
        # Already anonymized IDs (CUST_XXXXXX) are kept as-is
        if customer_id.startswith("CUST_"):
            return customer_id
            
        # Hash any other format
        hash_input = f"{customer_id}{self.salt}".encode()
        return f"ANON_{hashlib.sha256(hash_input).hexdigest()[:12]}"
    
    def generalize_age(self, age: int) -> str:
        """Generalize age into ranges for k-anonymity"""
        if age < 25:
            return "18-24"
        elif age < 35:
            return "25-34"
        elif age < 45:
            return "35-44"
        elif age < 55:
            return "45-54"
        elif age < 65:
            return "55-64"
        else:
            return "65+"
    
    def anonymize_location(self, location: str) -> str:
        """Generalize location to region level"""
        # Map specific locations to regions
        location_mapping = {
            "Sofia": "Western Bulgaria",
            "Plovdiv": "Southern Bulgaria",
            "Varna": "Eastern Bulgaria",
            "Burgas": "Eastern Bulgaria",
            # Add more mappings as needed
            # I am already working on this issue. All liste will be provided
            # from IT Department. 
        }
        return location_mapping.get(location, "Bulgaria")
    
    def add_temporal_noise(self, timestamp: datetime) -> datetime:
        """Add random noise to timestamps (±30 minutes)"""
        noise_minutes = np.random.randint(-30, 31)
        return timestamp + timedelta(minutes=noise_minutes)
    
    def anonymize_dataframe(self, df: pd.DataFrame, 
                          config: Dict[str, str]) -> pd.DataFrame:
        """
        Apply anonymization to entire dataframe based on config.
        
        Config example:
        {
            'customer_id': 'hash',
            'age': 'generalize', 
            'location': 'generalize',
            'timestamp': 'noise'
        }
        """
        df_anon = df.copy()
        
        for column, method in config.items():
            if column not in df_anon.columns:
                continue
                
            if method == 'hash':
                df_anon[column] = df_anon[column].apply(self.anonymize_customer_id)
            elif method == 'generalize' and column == 'age':
                df_anon[column] = df_anon[column].apply(self.generalize_age)
            elif method == 'generalize' and column == 'location':
                df_anon[column] = df_anon[column].apply(self.anonymize_location)
            elif method == 'noise' and column.endswith('timestamp'):
                df_anon[column] = df_anon[column].apply(self.add_temporal_noise)
            elif method == 'remove':
                df_anon = df_anon.drop(columns=[column])
                
        return df_anon