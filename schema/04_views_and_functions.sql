-- Function to calculate customer risk score güncelleyin:

CREATE OR REPLACE FUNCTION casino_data.calculate_risk_score(
    p_customer_id VARCHAR
) RETURNS DECIMAL AS $$
DECLARE
    v_risk_score DECIMAL(5,2) := 0;
    v_loss_rate DECIMAL(5,2);
    v_session_frequency DECIMAL(10,2);
    v_bet_acceleration DECIMAL(10,2);
BEGIN
    /*
    Academic Note: Risk score calculation is intentionally simplified 
    and designed for demonstration purposes. A production-ready score 
    would require validation by domain experts and consider additional 
    factors such as:
    - Time-based patterns (chase losses within sessions)
    - Bet size escalation patterns
    - Break-taking behavior
    - Multi-venue gambling indicators
    
    Current formula: Risk = (Loss Rate * 0.4) + (Session Frequency * 10)
    Capped at 100 for interpretability.
    */
    
    -- Calculate components
    SELECT 
        CASE WHEN SUM(total_bet) > 0 
             THEN (SUM(total_bet - total_win) / SUM(total_bet)) * 100 
             ELSE 0 END,
        COUNT(*)::DECIMAL / GREATEST(DATE_PART('day', MAX(session_start) - MIN(session_start)), 1)
    INTO v_loss_rate, v_session_frequency
    FROM casino_data.player_sessions
    WHERE customer_id = p_customer_id;
    
    -- Simple risk score calculation
    v_risk_score := LEAST(100, (v_loss_rate * 0.4) + (v_session_frequency * 10));
    
    -- Log calculation for academic audit
    INSERT INTO academic_audit.access_log 
    (student_id, ethics_ref, action, table_accessed, query_type)
    VALUES 
    ('mycc21', '10351-12382', 'RISK_CALCULATION', 'player_sessions', 'FUNCTION');
    
    RETURN v_risk_score;
END;
$$ LANGUAGE plpgsql;

COMMENT ON FUNCTION casino_data.calculate_risk_score IS 
'Simplified risk score for academic demonstration - see function body for limitations';