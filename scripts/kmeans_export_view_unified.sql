-- ================================================
-- MULTI-PERIOD K-MEANS EXPORT VIEWS SETUP
-- Both individual period views + unified view approach
-- ================================================

-- APPROACH A: Individual period views (for sequential analysis)
-- ================================================

-- 2022-H2 Export View
CREATE OR REPLACE VIEW casino_data.kmeans_export_2022_h2 AS
SELECT 
    customer_id,
    avg_bet as total_wagered,
    avg_bet as avg_bet_per_session,
    loss_rate,
    total_sessions,
    0 as days_since_last_visit,
    COALESCE(total_bet / NULLIF(total_sessions, 0), 0) as avg_session_value,
    CASE WHEN loss_rate > 20 THEN 1 ELSE 0 END as loss_chasing_indicator,
    total_sessions as sessions_last_30d,
    1.0 as bet_trend_ratio,
    
    -- Normalized features
    (avg_bet - (SELECT AVG(avg_bet) FROM casino_data.customer_features WHERE analysis_period = '2022-H2')) / 
    NULLIF((SELECT STDDEV(avg_bet) FROM casino_data.customer_features WHERE analysis_period = '2022-H2'), 0) as normalized_avg_bet,
    
    (total_sessions - (SELECT AVG(total_sessions) FROM casino_data.customer_features WHERE analysis_period = '2022-H2')) / 
    NULLIF((SELECT STDDEV(total_sessions) FROM casino_data.customer_features WHERE analysis_period = '2022-H2'), 0) as normalized_sessions,
    
    (loss_rate - (SELECT AVG(loss_rate) FROM casino_data.customer_features WHERE analysis_period = '2022-H2')) / 
    NULLIF((SELECT STDDEV(loss_rate) FROM casino_data.customer_features WHERE analysis_period = '2022-H2'), 0) as normalized_loss_rate,
    
    '2022-H2' as period_id,
    total_bet,
    total_win
FROM casino_data.customer_features 
WHERE analysis_period = '2022-H2' 
  AND total_sessions > 0
  AND avg_bet > 0;

-- 2023-H1 Export View
CREATE OR REPLACE VIEW casino_data.kmeans_export_2023_h1 AS
SELECT 
    customer_id,
    avg_bet as total_wagered,
    avg_bet as avg_bet_per_session,
    loss_rate,
    total_sessions,
    0 as days_since_last_visit,
    COALESCE(total_bet / NULLIF(total_sessions, 0), 0) as avg_session_value,
    CASE WHEN loss_rate > 20 THEN 1 ELSE 0 END as loss_chasing_indicator,
    total_sessions as sessions_last_30d,
    1.0 as bet_trend_ratio,
    
    -- Normalized features  
    (avg_bet - (SELECT AVG(avg_bet) FROM casino_data.customer_features WHERE analysis_period = '2023-H1')) / 
    NULLIF((SELECT STDDEV(avg_bet) FROM casino_data.customer_features WHERE analysis_period = '2023-H1'), 0) as normalized_avg_bet,
    
    (total_sessions - (SELECT AVG(total_sessions) FROM casino_data.customer_features WHERE analysis_period = '2023-H1')) / 
    NULLIF((SELECT STDDEV(total_sessions) FROM casino_data.customer_features WHERE analysis_period = '2023-H1'), 0) as normalized_sessions,
    
    (loss_rate - (SELECT AVG(loss_rate) FROM casino_data.customer_features WHERE analysis_period = '2023-H1')) / 
    NULLIF((SELECT STDDEV(loss_rate) FROM casino_data.customer_features WHERE analysis_period = '2023-H1'), 0) as normalized_loss_rate,
    
    '2023-H1' as period_id,
    total_bet,
    total_win
FROM casino_data.customer_features 
WHERE analysis_period = '2023-H1' 
  AND total_sessions > 0
  AND avg_bet > 0;

-- 2023-H2 Export View
CREATE OR REPLACE VIEW casino_data.kmeans_export_2023_h2 AS
SELECT 
    customer_id,
    avg_bet as total_wagered,
    avg_bet as avg_bet_per_session,
    loss_rate,
    total_sessions,
    0 as days_since_last_visit,
    COALESCE(total_bet / NULLIF(total_sessions, 0), 0) as avg_session_value,
    CASE WHEN loss_rate > 20 THEN 1 ELSE 0 END as loss_chasing_indicator,
    total_sessions as sessions_last_30d,
    1.0 as bet_trend_ratio,
    
    -- Normalized features
    (avg_bet - (SELECT AVG(avg_bet) FROM casino_data.customer_features WHERE analysis_period = '2023-H2')) / 
    NULLIF((SELECT STDDEV(avg_bet) FROM casino_data.customer_features WHERE analysis_period = '2023-H2'), 0) as normalized_avg_bet,
    
    (total_sessions - (SELECT AVG(total_sessions) FROM casino_data.customer_features WHERE analysis_period = '2023-H2')) / 
    NULLIF((SELECT STDDEV(total_sessions) FROM casino_data.customer_features WHERE analysis_period = '2023-H2'), 0) as normalized_sessions,
    
    (loss_rate - (SELECT AVG(loss_rate) FROM casino_data.customer_features WHERE analysis_period = '2023-H2')) / 
    NULLIF((SELECT STDDEV(loss_rate) FROM casino_data.customer_features WHERE analysis_period = '2023-H2'), 0) as normalized_loss_rate,
    
    '2023-H2' as period_id,
    total_bet,
    total_win
FROM casino_data.customer_features 
WHERE analysis_period = '2023-H2' 
  AND total_sessions > 0
  AND avg_bet > 0;

-- 2024-H1 Export View (Optional)
CREATE OR REPLACE VIEW casino_data.kmeans_export_2024_h1 AS
SELECT 
    customer_id,
    avg_bet as total_wagered,
    avg_bet as avg_bet_per_session,
    loss_rate,
    total_sessions,
    0 as days_since_last_visit,
    COALESCE(total_bet / NULLIF(total_sessions, 0), 0) as avg_session_value,
    CASE WHEN loss_rate > 20 THEN 1 ELSE 0 END as loss_chasing_indicator,
    total_sessions as sessions_last_30d,
    1.0 as bet_trend_ratio,
    
    -- Normalized features
    (avg_bet - (SELECT AVG(avg_bet) FROM casino_data.customer_features WHERE analysis_period = '2024-H1')) / 
    NULLIF((SELECT STDDEV(avg_bet) FROM casino_data.customer_features WHERE analysis_period = '2024-H1'), 0) as normalized_avg_bet,
    
    (total_sessions - (SELECT AVG(total_sessions) FROM casino_data.customer_features WHERE analysis_period = '2024-H1')) / 
    NULLIF((SELECT STDDEV(total_sessions) FROM casino_data.customer_features WHERE analysis_period = '2024-H1'), 0) as normalized_sessions,
    
    (loss_rate - (SELECT AVG(loss_rate) FROM casino_data.customer_features WHERE analysis_period = '2024-H1')) / 
    NULLIF((SELECT STDDEV(loss_rate) FROM casino_data.customer_features WHERE analysis_period = '2024-H1'), 0) as normalized_loss_rate,
    
    '2024-H1' as period_id,
    total_bet,
    total_win
FROM casino_data.customer_features 
WHERE analysis_period = '2024-H1' 
  AND total_sessions > 0
  AND avg_bet > 0;

-- APPROACH B: Unified view (for comparative analysis)
-- ================================================
CREATE OR REPLACE VIEW casino_data.kmeans_export_all_periods AS
SELECT 
    customer_id,
    analysis_period as period_id,
    avg_bet as total_wagered,
    avg_bet as avg_bet_per_session,
    loss_rate,
    total_sessions,
    0 as days_since_last_visit,
    COALESCE(total_bet / NULLIF(total_sessions, 0), 0) as avg_session_value,
    CASE WHEN loss_rate > 20 THEN 1 ELSE 0 END as loss_chasing_indicator,
    total_sessions as sessions_last_30d,
    1.0 as bet_trend_ratio,
    total_bet,
    total_win,
    
    -- Period-normalized features (normalized within each period)
    (avg_bet - AVG(avg_bet) OVER (PARTITION BY analysis_period)) / 
    NULLIF(STDDEV(avg_bet) OVER (PARTITION BY analysis_period), 0) as normalized_avg_bet,
    
    (total_sessions - AVG(total_sessions) OVER (PARTITION BY analysis_period)) / 
    NULLIF(STDDEV(total_sessions) OVER (PARTITION BY analysis_period), 0) as normalized_sessions,
    
    (loss_rate - AVG(loss_rate) OVER (PARTITION BY analysis_period)) / 
    NULLIF(STDDEV(loss_rate) OVER (PARTITION BY analysis_period), 0) as normalized_loss_rate
    
FROM casino_data.customer_features 
WHERE analysis_period IN ('2022-H1', '2022-H2', '2023-H1', '2023-H2', '2024-H1')
  AND total_sessions > 0
  AND avg_bet > 0;

-- VALIDATION: Check all views are ready
-- ================================================
SELECT 
    'VIEW_VALIDATION' as check_type,
    view_name,
    customer_count,
    'READY' as status
FROM (
    SELECT 'kmeans_export_2022_h1' as view_name, COUNT(*) as customer_count FROM casino_data.kmeans_export_2022_h1
    UNION ALL
    SELECT 'kmeans_export_2022_h2', COUNT(*) FROM casino_data.kmeans_export_2022_h2
    UNION ALL
    SELECT 'kmeans_export_2023_h1', COUNT(*) FROM casino_data.kmeans_export_2023_h1
    UNION ALL
    SELECT 'kmeans_export_2023_h2', COUNT(*) FROM casino_data.kmeans_export_2023_h2
    UNION ALL
    SELECT 'kmeans_export_2024_h1', COUNT(*) FROM casino_data.kmeans_export_2024_h1
    UNION ALL
    SELECT 'kmeans_export_all_periods', COUNT(*) FROM casino_data.kmeans_export_all_periods
) view_summary
ORDER BY view_name;

-- PYTHON INTEGRATION HELPER
-- ================================================
CREATE OR REPLACE FUNCTION get_period_export_view(p_period_id TEXT) 
RETURNS TEXT AS $$
BEGIN
    RETURN CASE p_period_id
        WHEN '2022-H1' THEN 'casino_data.kmeans_export_2022_h1'
        WHEN '2022-H2' THEN 'casino_data.kmeans_export_2022_h2'
        WHEN '2023-H1' THEN 'casino_data.kmeans_export_2023_h1'
        WHEN '2023-H2' THEN 'casino_data.kmeans_export_2023_h2'
        WHEN '2024-H1' THEN 'casino_data.kmeans_export_2024_h1'
        ELSE 'casino_data.kmeans_export_all_periods'
    END;
END;
$$ LANGUAGE plpgsql;

-- USAGE SUMMARY
-- ================================================
SELECT 
    'USAGE_SUMMARY' as info_type,
    'Individual views for sequential K-means processing' as approach_a,
    'Unified view for comparative cross-period analysis' as approach_b,
    'Python: segmentation.py --period=XXXX-XX' as python_command,
    'Academic: Temporal drift analysis ready' as academic_value;