-- schema/02_academic_audit_tables.sql
-- Academic Compliance and Audit Trail Tables
-- Ethics Ref: 10351-12382

-- Academic compliance tracking
CREATE TABLE academic_audit.compliance_metadata (
    compliance_id SERIAL PRIMARY KEY,
    ethics_approval_ref VARCHAR(50) DEFAULT '10351-12382',
    student_id VARCHAR(50) DEFAULT 'mycc21',
    student_name VARCHAR(100) DEFAULT 'Muhammed Yavuzhan CANLI',
    supervisor VARCHAR(100) DEFAULT 'Dr. Moody Alam',
    institution VARCHAR(100) DEFAULT 'University of Bath',
    department VARCHAR(200) DEFAULT 'Computer Science - Software Engineering',
    project_title TEXT DEFAULT 'Casino Customer Segmentation and Promotional Decision Framework',
    data_classification VARCHAR(50) DEFAULT 'ANONYMIZED_BUSINESS_DATA',
    gdpr_compliance VARCHAR(100) DEFAULT 'Article 26 Full Anonymization',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Data access logging for ethics compliance
CREATE TABLE academic_audit.access_log (
    log_id BIGSERIAL PRIMARY KEY,
    student_id VARCHAR(50),
    ethics_ref VARCHAR(50),
    action VARCHAR(100),
    table_accessed VARCHAR(100),
    query_type VARCHAR(50),
    record_count INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    session_id UUID DEFAULT gen_random_uuid()
);

-- Data provenance tracking
CREATE TABLE academic_audit.data_provenance (
    provenance_id BIGSERIAL PRIMARY KEY,
    original_source VARCHAR(200),
    original_table VARCHAR(100),
    target_table VARCHAR(100),
    anonymization_method TEXT,
    transformation_applied TEXT,
    academic_purpose TEXT,
    records_affected INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'mycc21'
);

-- Model versioning for reproducibility
CREATE TABLE academic_audit.model_registry (
    model_id SERIAL PRIMARY KEY,
    model_name VARCHAR(100),
    model_type VARCHAR(50), -- 'segmentation' or 'promotion'
    model_version VARCHAR(20),
    training_date TIMESTAMP,
    performance_metrics JSONB,
    feature_list TEXT[],
    hyperparameters JSONB,
    file_path VARCHAR(500),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial compliance record
INSERT INTO academic_audit.compliance_metadata 
(ethics_approval_ref, student_id, student_name, supervisor, institution)
VALUES 
('10351-12382', 'mycc21', 'Muhammed Yavuzhan CANLI', 'Dr. Moody Alam', 'University of Bath');