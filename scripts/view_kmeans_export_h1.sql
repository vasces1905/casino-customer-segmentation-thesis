CREATE OR REPLACE VIEW casino_data.kmeans_export_2021_h1 AS
SELECT 
    customer_id,
    avg_bet_period as total_wagered,
    avg_bet_period,
    loss_rate_period as loss_rate,
    total_sessions_period as total_sessions,
    0 as days_since_last_visit,
    COALESCE(total_bet_period / NULLIF(total_sessions_period, 0), 0) as avg_session_value,
    CASE WHEN loss_rate_period > 60 THEN 1 ELSE 0 END as loss_chasing_indicator,
    total_sessions_period as sessions_last_30d,
    1.0 as bet_trend_ratio
FROM casino_data.temporal_segments 
WHERE period_id = '2021-H1' 
  AND total_sessions_period > 0;
