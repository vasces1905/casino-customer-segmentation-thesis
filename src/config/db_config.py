# src/config/db_config.py

"""
Database Configuration
======================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution
"""

""" 
Technical Explanation & What are included
- Holds the parameters used to connect to PostgreSQL.
- Reads values from environment variables (.env).
- Specifies the mapping of tables and audit schema.
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

# Academic audit schema - Schema to write access logs
AUDIT_SCHEMA = "academic_audit"
MAIN_SCHEMA = "casino_data"

# Table name mapping (for compatibility layer) - Real system equivalents of Casino-1 (simulation) table names
TABLE_MAPPING = {
    "customers": "customer_demographics",
    "sessions": "player_sessions", 
    "transactions": "player_transactions",
    "promotions": "promotion_history"
}