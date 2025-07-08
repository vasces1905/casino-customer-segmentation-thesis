-- schema/01_create_database.sql
-- University of Bath - Casino Customer Segmentation Research
-- Student: Muhammed Yavuzhan CANLI
-- Ethics Approval: 10351-12382
-- Academic database initialization (Docker PostgreSQL)

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- For gen_random_uuid()
CREATE EXTENSION IF NOT EXISTS tablefunc; -- For crosstab queries (optional)

-- Create schemas
CREATE SCHEMA IF NOT EXISTS casino_data;
CREATE SCHEMA IF NOT EXISTS academic_audit;

-- Set search path
SET search_path TO casino_data, academic_audit, public;

-- Grant permissions to researcher user
GRANT ALL PRIVILEGES ON SCHEMA casino_data TO researcher;
GRANT ALL PRIVILEGES ON SCHEMA academic_audit TO researcher;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA casino_data TO researcher;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA academic_audit TO researcher;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA casino_data TO researcher;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA academic_audit TO researcher;

-- Academic compliance message
DO $$
BEGIN
    RAISE NOTICE '=== BATH UNIVERSITY ACADEMIC DATABASE ===';
    RAISE NOTICE 'Student: Muhammed Yavuzhan CANLI';
    RAISE NOTICE 'Ethics Approval: 10351-12382';
    RAISE NOTICE 'Schemas created: casino_data, academic_audit';
    RAISE NOTICE '=========================================';
END $$;