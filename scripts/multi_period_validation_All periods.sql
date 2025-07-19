-- ================================================
-- MULTI-PERIOD VALIDATION - ALL PERIODS STATUS
-- Validating temporal segmentation across 2021-2023
-- ================================================

-- CHECK 1: Period coverage summary
-- ================================================
SELECT 
    'PERIOD_COVERAGE_SUMMARY' as analysis_type,
    analysis_period,
    COUNT(*) as customers_with_features,
    ROUND(AVG(avg_bet)::numeric, 2) as avg_bet_period,
    ROUND(AVG(total_sessions)::numeric, 1) as avg_sessions_period,
    ROUND(AVG(loss_rate)::numeric, 2) as avg_loss_rate_period,
    MIN(feature_created_at)::date as first_feature_date,
    MAX(feature_created_at)::date as last_feature_date
FROM casino_data.customer_features 
WHERE analysis_period IS NOT NULL
GROUP BY analysis_period
ORDER BY analysis_period;

-- CHECK 2: Customer overlap across periods (cohort analysis)
-- ================================================
WITH period_customers AS (
    SELECT 
        customer_id,
        analysis_period,
        avg_bet,
        total_sessions
    FROM casino_data.customer_features 
    WHERE analysis_period IS NOT NULL
),
customer_period_counts AS (
    SELECT 
        customer_id,
        COUNT(DISTINCT analysis_period) as periods_active,
        STRING_AGG(analysis_period, ', ' ORDER BY analysis_period) as active_periods
    FROM period_customers
    GROUP BY customer_id
)
SELECT 
    'CUSTOMER_COHORT_ANALYSIS' as analysis_type,
    periods_active,
    COUNT(*) as customer_count,
    ROUND((COUNT(*) * 100.0 / SUM(COUNT(*)) OVER()), 2) as percentage,
    CASE 
        WHEN periods_active = 1 THEN 'Single_Period'
        WHEN periods_active = 2 THEN 'Two_Periods' 
        WHEN periods_active = 3 THEN 'Three_Periods'
        WHEN periods_active >= 4 THEN 'Multi_Period_Loyal'
    END as customer_type
FROM customer_period_counts
GROUP BY periods_active
ORDER BY periods_active;

-- CHECK 3: Temporal trends analysis
-- ================================================
SELECT 
    'TEMPORAL_TRENDS' as analysis_type,
    analysis_period,
    COUNT(*) as active_customers,
    
    -- Customer activity levels
    COUNT(CASE WHEN total_sessions > 20 THEN 1 END) as high_activity_customers,
    COUNT(CASE WHEN total_sessions BETWEEN 5 AND 20 THEN 1 END) as medium_activity_customers,
    COUNT(CASE WHEN total_sessions < 5 THEN 1 END) as low_activity_customers,
    
    -- Betting levels  
    COUNT(CASE WHEN avg_bet > 5000 THEN 1 END) as high_bet_customers,
    COUNT(CASE WHEN avg_bet BETWEEN 1500 AND 5000 THEN 1 END) as medium_bet_customers,
    COUNT(CASE WHEN avg_bet < 1500 THEN 1 END) as low_bet_customers,
    
    -- Business metrics
    ROUND(SUM(total_bet)::numeric, 2) as period_total_volume,
    ROUND(AVG(loss_rate)::numeric, 2) as period_avg_loss_rate
FROM casino_data.customer_features 
WHERE analysis_period IS NOT NULL
GROUP BY analysis_period
ORDER BY analysis_period;

-- CHECK 4: Segmentation readiness for each period
-- ================================================
SELECT 
    'SEGMENTATION_READINESS' as analysis_type,
    analysis_period,
    COUNT(*) as total_customers,
    COUNT(CASE WHEN avg_bet > 0 AND total_sessions > 0 THEN 1 END) as kmeans_ready_customers,
    ROUND(
        COUNT(CASE WHEN avg_bet > 0 AND total_sessions > 0 THEN 1 END) * 100.0 / COUNT(*), 2
    ) as readiness_percentage,
    CASE 
        WHEN COUNT(CASE WHEN avg_bet > 0 AND total_sessions > 0 THEN 1 END) > 1000 
        THEN 'READY_FOR_KMEANS'
        ELSE 'INSUFFICIENT_DATA'
    END as kmeans_status
FROM casino_data.customer_features 
WHERE analysis_period IS NOT NULL
GROUP BY analysis_period
ORDER BY analysis_period;

-- CHECK 5: Rule-based segmentation status
-- ================================================
SELECT 
    'RULE_BASED_SEGMENTS_STATUS' as analysis_type,
    ts.period_id,
    COUNT(*) as customers_segmented,
    COUNT(DISTINCT ts.segment_id) as segments_created,
    ROUND(AVG(ts.avg_bet_period)::numeric, 2) as avg_bet_segmented,
    MAX(ts.segment_created_at)::date as last_segmentation_date,
    CASE 
        WHEN COUNT(*) > 0 THEN 'SEGMENTS_EXIST'
        ELSE 'NO_SEGMENTS'
    END as segment_status
FROM casino_data.temporal_segments ts
GROUP BY ts.period_id
ORDER BY ts.period_id;

-- CHECK 6: Data quality assessment
-- ================================================
SELECT 
    'DATA_QUALITY_ASSESSMENT' as analysis_type,
    analysis_period,
    COUNT(*) as total_records,
    COUNT(CASE WHEN avg_bet IS NULL THEN 1 END) as null_avg_bet,
    COUNT(CASE WHEN total_sessions IS NULL THEN 1 END) as null_sessions,
    COUNT(CASE WHEN loss_rate IS NULL THEN 1 END) as null_loss_rate,
    COUNT(CASE WHEN avg_bet <= 0 THEN 1 END) as zero_or_negative_bet,
    COUNT(CASE WHEN total_sessions <= 0 THEN 1 END) as zero_sessions,
    ROUND(
        (COUNT(*) - COUNT(CASE WHEN avg_bet IS NULL OR total_sessions IS NULL OR avg_bet <= 0 THEN 1 END)) * 100.0 / COUNT(*), 2
    ) as data_quality_percentage
FROM casino_data.customer_features 
WHERE analysis_period IS NOT NULL
GROUP BY analysis_period
ORDER BY analysis_period;

-- CHECK 7: K-means export views readiness
-- ================================================
SELECT 
    'KMEANS_EXPORT_READINESS' as analysis_type,
    '2021-H1' as period_id,
    COUNT(*) as customers_in_export_view,
    'casino_data.kmeans_export_2021_h1' as view_name,
    'BASELINE_READY' as status
FROM casino_data.kmeans_export_2021_h1;

-- CHECK 8: Next steps recommendation
-- ================================================
SELECT 
    'NEXT_STEPS_RECOMMENDATION' as analysis_type,
    'All periods have feature data calculated' as current_status,
    'Ready for sequential K-means segmentation' as next_action,
    'Start with: Python segmentation.py --period=2021-H1' as first_command,
    'Then proceed: 2022-H1, 2022-H2, 2023-H1, 2023-H2' as sequence,
    'Academic goal: Temporal behavior evolution analysis' as objective;