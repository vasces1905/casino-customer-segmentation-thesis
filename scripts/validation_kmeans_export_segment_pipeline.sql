-- ================================================
-- DUAL SEGMENTATION SETUP: RULE-BASED vs K-MEANS
-- Academic comparative study implementation
-- ================================================

-- CREATE separate table for K-means results
-- ================================================
CREATE TABLE IF NOT EXISTS casino_data.kmeans_segments (
    customer_id TEXT,
    period_id TEXT,
    cluster_id INTEGER,
    cluster_label TEXT,
    silhouette_score NUMERIC,
    distance_to_centroid NUMERIC,
    normalized_features JSONB,
    kmeans_version TEXT DEFAULT 'sklearn_v1',
    model_metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id, period_id, kmeans_version)
);

-- CREATE comparison analysis table
-- ================================================
CREATE TABLE IF NOT EXISTS casino_data.segmentation_comparison (
    customer_id TEXT,
    period_id TEXT,
    rule_based_segment_id INTEGER,
    rule_based_segment_name TEXT,
    kmeans_cluster_id INTEGER,
    kmeans_cluster_label TEXT,
    agreement_status TEXT, -- 'match', 'partial_match', 'mismatch'
    comparison_notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (customer_id, period_id)
);

-- CONTINUE with rule-based segmentation (Track 1)
-- ================================================
-- The previous INSERT for temporal_segments is CORRECT and NEEDED
-- This provides the business logic baseline

-- VERIFICATION: Check if rule-based segmentation completed
-- ================================================
SELECT 
    'RULE_BASED_TRACK_STATUS' as track_type,
    COUNT(*) as customers_segmented,
    COUNT(DISTINCT segment_id) as segments_created,
    MIN(segment_created_at) as first_segment_time,
    MAX(segment_created_at) as last_segment_time,
    'Track 1: Business Rules' as description
FROM casino_data.temporal_segments
WHERE period_id = '2021-H1';

-- PREPARE data export for Python K-means (Track 2)
-- ================================================
-- Enhanced export view with more ML features
CREATE OR REPLACE VIEW casino_data.kmeans_export_2021_h1_enhanced AS
SELECT 
    customer_id,
    
    -- Core features for K-means
    avg_bet_period as avg_bet,
    total_sessions_period as total_sessions,
    loss_rate_period as loss_rate,
    total_bet_period as total_wagered,
    
    -- Derived features
    COALESCE(total_bet_period / NULLIF(total_sessions_period, 0), 0) as avg_session_value,
    CASE 
        WHEN loss_rate_period > 60 THEN 1 
        ELSE 0 
    END as high_loss_indicator,
    
    -- Normalized features (for better K-means performance)
    (avg_bet_period - (SELECT AVG(avg_bet_period) FROM casino_data.temporal_segments WHERE period_id = '2021-H1')) / 
    NULLIF((SELECT STDDEV(avg_bet_period) FROM casino_data.temporal_segments WHERE period_id = '2021-H1'), 0) as normalized_avg_bet,
    
    (total_sessions_period - (SELECT AVG(total_sessions_period) FROM casino_data.temporal_segments WHERE period_id = '2021-H1')) / 
    NULLIF((SELECT STDDEV(total_sessions_period) FROM casino_data.temporal_segments WHERE period_id = '2021-H1'), 0) as normalized_sessions,
    
    (loss_rate_period - (SELECT AVG(loss_rate_period) FROM casino_data.temporal_segments WHERE period_id = '2021-H1')) / 
    NULLIF((SELECT STDDEV(loss_rate_period) FROM casino_data.temporal_segments WHERE period_id = '2021-H1'), 0) as normalized_loss_rate,
    
    -- Meta information
    '2021-H1' as period_id,
    'baseline' as period_type
    
FROM casino_data.temporal_segments 
WHERE period_id = '2021-H1' 
  AND total_sessions_period > 0
  AND avg_bet_period > 0;

-- VALIDATION: Python export readiness
-- ================================================
SELECT 
    'PYTHON_KMEANS_EXPORT_READY' as status,
    COUNT(*) as customers_ready,
    COUNT(CASE WHEN normalized_avg_bet IS NOT NULL THEN 1 END) as customers_with_normalized_features,
    ROUND(AVG(avg_bet)::numeric, 2) as avg_bet_in_export,
    ROUND(AVG(total_sessions)::numeric, 1) as avg_sessions_in_export,
    ROUND(AVG(loss_rate)::numeric, 2) as avg_loss_rate_in_export,
    'View: casino_data.kmeans_export_2021_h1_enhanced' as export_view,
    'Next: Python segmentation.py --period=2021-H1' as next_command
FROM casino_data.kmeans_export_2021_h1_enhanced;

-- CREATE Python integration helper function
-- ================================================
CREATE OR REPLACE FUNCTION insert_kmeans_results(
    p_customer_data JSONB,
    p_period_id TEXT,
    p_model_metadata JSONB DEFAULT '{}'::JSONB
) RETURNS INTEGER AS $$
DECLARE
    inserted_count INTEGER := 0;
    customer_record JSONB;
BEGIN
    -- Insert K-means results from Python
    FOR customer_record IN SELECT * FROM jsonb_array_elements(p_customer_data)
    LOOP
        INSERT INTO casino_data.kmeans_segments (
            customer_id,
            period_id,
            cluster_id,
            cluster_label,
            silhouette_score,
            distance_to_centroid,
            model_metadata
        ) VALUES (
            customer_record->>'customer_id',
            p_period_id,
            (customer_record->>'cluster_id')::INTEGER,
            customer_record->>'cluster_label',
            COALESCE((customer_record->>'silhouette_score')::NUMERIC, 0),
            COALESCE((customer_record->>'distance_to_centroid')::NUMERIC, 0),
            p_model_metadata
        )
        ON CONFLICT (customer_id, period_id, kmeans_version) 
        DO UPDATE SET
            cluster_id = EXCLUDED.cluster_id,
            cluster_label = EXCLUDED.cluster_label,
            silhouette_score = EXCLUDED.silhouette_score,
            distance_to_centroid = EXCLUDED.distance_to_centroid,
            model_metadata = EXCLUDED.model_metadata,
            created_at = CURRENT_TIMESTAMP;
        
        inserted_count := inserted_count + 1;
    END LOOP;
    
    RETURN inserted_count;
END;
$$ LANGUAGE plpgsql;

-- SUMMARY: Current status
-- ================================================
SELECT 
    'DUAL_SEGMENTATION_SETUP_COMPLETE' as status,
    'Track 1: Rule-based segmentation in temporal_segments' as track1_status,
    'Track 2: K-means segmentation ready for Python' as track2_status,
    'Academic Value: Comparative analysis of two approaches' as academic_contribution,
    COUNT(*) as baseline_customers_ready
FROM casino_data.kmeans_export_2021_h1_enhanced;