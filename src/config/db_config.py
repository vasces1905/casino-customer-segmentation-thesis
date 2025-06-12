# src/config/db_config.py

"""
Database Configuration
======================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution
"""

import os
from typing import Dict

# Database connection parameters (will use environment variables)
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5432"),
    "database": os.getenv("DB_NAME", "casino_research"),
    "user": os.getenv("DB_USER", "researcher"),
    "password": os.getenv("DB_PASSWORD", "")  # Will be set via .env
}

# Academic audit schema
AUDIT_SCHEMA = "academic_audit"
MAIN_SCHEMA = "casino_data"

# Table name mapping (for compatibility layer)
TABLE_MAPPING = {
    "customers": "customer_demographics",
    "sessions": "player_sessions", 
    "transactions": "player_transactions",
    "promotions": "promotion_history"
}