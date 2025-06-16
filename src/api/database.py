# src/api/database.py

"""
Database Dependencies for FastAPI
=================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
"""

from typing import Generator
import logging
from contextlib import contextmanager

from ..data.db_connector import AcademicDBConnector

logger = logging.getLogger(__name__)

# Database connector instance
db_connector = AcademicDBConnector()

def get_db_connection():
    """
    Dependency to get database connection.
    Ensures proper cleanup and audit logging.
    """
    with db_connector.get_connection() as conn:
        try:
            yield conn
        finally:
            # Audit logging handled by connector
            pass

# Test database connection on import
try:
    if db_connector.test_connection():
        logger.info("Database connection successful")
    else:
        logger.warning("Database connection failed - API will run in mock mode")
except Exception as e:
    logger.warning(f"Database not available: {e} - Running in mock mode")