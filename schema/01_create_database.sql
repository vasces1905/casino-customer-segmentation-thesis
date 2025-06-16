-- schema/01_create_database.sql
-- University of Bath - Casino Customer Segmentation Research
-- Student: Muhammed Yavuzhan CANLI
-- Ethics Approval: 10351-12382

-- Create database (run as superuser)
CREATE DATABASE casino_research
    WITH 
    OWNER = postgres
    ENCODING = 'UTF8'
    CONNECTION LIMIT = -1;

-- Connect to the new database
\c casino_research;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- For gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS tablefunc; -- For crosstab queries (optional)

-- Create schemas
CREATE SCHEMA IF NOT EXISTS casino_data;
CREATE SCHEMA IF NOT EXISTS academic_audit;

-- Set search path
SET search_path TO casino_data, academic_audit, public;

-- Grant permissions (adjust username as needed)
GRANT ALL PRIVILEGES ON SCHEMA casino_data TO researcher;
GRANT ALL PRIVILEGES ON SCHEMA academic_audit TO researcher;
📝 ADIM 24: schema/02_academic_audit_table