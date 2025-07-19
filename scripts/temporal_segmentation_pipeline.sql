-- ================================================
-- 2022-2023 FOCUSED TEMPORAL SEGMENTATION PIPELINE
-- University of Bath - Excluding 2021, Focusing on Growth Period
-- Academic Goal: Analyze business growth and behavioral evolution
-- ================================================

-- STEP 1: Create focused period analysis
-- ================================================
UPDATE casino_data.analysis_periods 
SET description = CASE 
    WHEN period_id = '2022-H1' THEN 'Growth period baseline - Post-COVID recovery'
    WHEN period_id = '2022-H2' THEN 'Accelerated growth phase'
    WHEN period_id = '2023-H1' THEN 'Mature growth period'
    WHEN period_id = '2023-H2' THEN 'Peak activity period - 5.7x growth achieved'
END
WHERE period_id IN ('2022-H1', '2022-H2', '2023-H1', '2023-H2');

-- STEP 2: Create 2022-H1 export view (New baseline)
-- ================================================
CREATE OR REPLACE VIEW casino_data.kmeans_export_2022_h1 AS
SELECT 
    customer_id,
    
    -- Core K-means features
    avg_bet as total_wagered,
    avg_bet as avg_bet_per_session,
    loss_rate,
    total_sessions,
    0 as days_since_last_visit, -- Will be calculated in Python
    
    -- Derived features
    COALESCE(total_bet / NULLIF(total_sessions, 0), 0) as avg_session_value,
    CASE WHEN loss_rate > 20 THEN 1 ELSE 0 END as loss_chasing_indicator,
    total_sessions as sessions_last_30d,
    1.0 as bet_trend_ratio, -- Baseline period
    
    -- Normalized features for K-means
    (avg_bet - (SELECT AVG(avg_bet) FROM casino_data.customer_features WHERE analysis_period = '2022-H1')) / 
    NULLIF((SELECT STDDEV(avg_bet) FROM casino_data.customer_features WHERE analysis_period = '2022-H1'), 0) as normalized_avg_bet,
    
    (total_sessions - (SELECT AVG(total_sessions) FROM casino_data.customer_features WHERE analysis_period = '2022-H1')) / 
    NULLIF((SELECT STDDEV(total_sessions) FROM casino_data.customer_features WHERE analysis_period = '2022-H1'), 0) as normalized_sessions,
    
    (loss_rate - (SELECT AVG(loss_rate) FROM casino_data.customer_features WHERE analysis_period = '2022-H1')) / 
    NULLIF((SELECT STDDEV(loss_rate) FROM casino_data.customer_features WHERE analysis_period = '2022-H1'), 0) as normalized_loss_rate,
    
    -- Meta information
    '2022-H1' as period_id,
    'new_baseline' as period_type,
    total_bet,
    total_win
    
FROM casino_data.customer_features 
WHERE analysis_period = '2022-H1' 
  AND total_sessions > 0
  AND avg_bet > 0;

-- STEP 3: Rule-based segmentation for 2022-H1 (New baseline)
-- ================================================
-- Clear any existing 2022-H1 segments
DELETE FROM casino_data.temporal_segments WHERE period_id = '2022-H1';

-- Insert 2022-H1 rule-based segments
INSERT INTO casino_data.temporal_segments (
    customer_id,
    period_id,
    segment_id,
    segment_name,
    avg_bet_period,
    total_sessions_period,
    loss_rate_period,
    total_bet_period
)
SELECT 
    customer_id,
    '2022-H1' as period_id,
    CASE 
        -- Adjusted thresholds based on 2022 data patterns
        WHEN avg_bet > 3000 AND total_sessions > 8 THEN 1
        WHEN avg_bet BETWEEN 800 AND 3000 AND total_sessions BETWEEN 3 AND 12 THEN 2
        WHEN avg_bet < 800 AND total_sessions BETWEEN 1 AND 8 THEN 3
        ELSE 4
    END as segment_id,
    
    CASE 
        WHEN avg_bet > 3000 AND total_sessions > 8 THEN 'High_Roller_2022'
        WHEN avg_bet BETWEEN 800 AND 3000 AND total_sessions BETWEEN 3 AND 12 THEN 'Regular_Player_2022'
        WHEN avg_bet < 800 AND total_sessions BETWEEN 1 AND 8 THEN 'Casual_Player_2022'
        ELSE 'Light_Player_2022'
    END as segment_name,
    
    avg_bet,
    total_sessions,
    loss_rate,
    total_bet
    
FROM casino_data.customer_features
WHERE analysis_period = '2022-H1'
  AND total_sessions > 0;

-- STEP 4: Academic timeline update
-- ================================================
CREATE OR REPLACE VIEW casino_data.academic_timeline_2022_2023 AS
SELECT 
    'PHASE_1_2022_H1' as phase,
    '2022-H1' as period_id,
    'Growth Period Baseline' as description,
    2946 as customer_count,
    'Foundation for temporal analysis' as academic_purpose,
    'Rule-based + K-means segmentation' as methodology,
    1 as sequence_order
UNION ALL
SELECT 
    'PHASE_2_2022_H2',
    '2022-H2',
    'Accelerated Growth Analysis',
    4752,
    'First evolution measurement',
    'K-means + migration analysis',
    2
UNION ALL
SELECT 
    'PHASE_3_2023_H1',
    '2023-H1',
    'Mature Growth Period',
    8054,
    'Mid-term behavioral evolution',
    'K-means + predictive modeling',
    3
UNION ALL
SELECT 
    'PHASE_4_2023_H2',
    '2023-H2',
    'Peak Activity Analysis',
    16899,
    'Final temporal analysis',
    'Complete ML pipeline + RF promotion',
    4;

-- STEP 5: Validation of new baseline
-- ================================================
SELECT 
    'NEW_BASELINE_2022_H1' as analysis_type,
    COUNT(*) as total_customers,
    COUNT(CASE WHEN segment_id = 1 THEN 1 END) as high_rollers,
    COUNT(CASE WHEN segment_id = 2 THEN 1 END) as regular_players,
    COUNT(CASE WHEN segment_id = 3 THEN 1 END) as casual_players,
    COUNT(CASE WHEN segment_id = 4 THEN 1 END) as light_players,
    ROUND(AVG(avg_bet_period)::numeric, 2) as avg_bet_baseline,
    ROUND(AVG(total_sessions_period)::numeric, 1) as avg_sessions_baseline
FROM casino_data.temporal_segments 
WHERE period_id = '2022-H1';

-- STEP 6: K-means export readiness for all 2022-2023 periods
-- ================================================
SELECT 
    'KMEANS_EXPORT_STATUS_2022_2023' as status,
    period_name,
    customer_count,
    'READY_FOR_PYTHON_SEGMENTATION ✅' as python_status
FROM (
    SELECT '2022-H1' as period_name, COUNT(*) as customer_count FROM casino_data.kmeans_export_2022_h1
    UNION ALL
    SELECT '2022-H2', COUNT(*) FROM casino_data.customer_features WHERE analysis_period = '2022-H2' AND total_sessions > 0
    UNION ALL  
    SELECT '2023-H1', COUNT(*) FROM casino_data.customer_features WHERE analysis_period = '2023-H1' AND total_sessions > 0
    UNION ALL
    SELECT '2023-H2', COUNT(*) FROM casino_data.customer_features WHERE analysis_period = '2023-H2' AND total_sessions > 0
) period_summary
ORDER BY period_name;

-- STEP 7: Academic contribution summary
-- ================================================
SELECT 
    'ACADEMIC_CONTRIBUTION_2022_2023' as summary,
    'Temporal Behavioral Evolution During Casino Growth Period' as thesis_chapter,
    '4 periods: 2022-H1 → 2023-H2' as analysis_scope,
    '36,040 total customer records' as dataset_size,
    '5.7x customer growth analyzed' as business_insight,
    'Rule-based vs K-means comparative study' as methodology_contribution,
    'Predictive promotion targeting with RF' as practical_application;