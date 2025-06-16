-- schema/03_casino_data_tables.sql
-- Main Casino Data Tables (Anonymized)
-- All personal data pre-anonymized per GDPR Article 26

-- Customer demographics (fully anonymized)
CREATE TABLE casino_data.customer_demographics (
    customer_id VARCHAR(20) PRIMARY KEY, -- Already anonymized (CUST_XXXXXX) after taken from Casino DB
    age_range VARCHAR(10), -- Generalized: '18-24', '25-34', etc.
    gender VARCHAR(10), -- M/F/Other
    region VARCHAR(50), -- Generalized location
    registration_month VARCHAR(7), -- YYYY-MM only
    customer_segment INTEGER, -- From ML clustering
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Player sessions (temporal noise added)
CREATE TABLE casino_data.player_sessions (
    session_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(20) REFERENCES casino_data.customer_demographics(customer_id),
    session_start TIMESTAMP, -- ±30 min noise added
    session_end TIMESTAMP,
    total_bet DECIMAL(10,2),
    total_win DECIMAL(10,2),
    game_type VARCHAR(50),
    machine_id VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Aggregated daily statistics (for performance)
CREATE TABLE casino_data.daily_aggregates (
    aggregate_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(20) REFERENCES casino_data.customer_demographics(customer_id),
    date DATE,
    total_sessions INTEGER,
    total_bet DECIMAL(10,2),
    total_win DECIMAL(10,2),
    unique_games INTEGER,
    avg_session_duration INTEGER, -- minutes
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(customer_id, date)
);

-- Machine/Zone mapping
CREATE TABLE casino_data.casino_machines (
    machine_id VARCHAR(20) PRIMARY KEY,
    machine_type VARCHAR(50),
    machine_zone VARCHAR(50),
    floor_location VARCHAR(20)
);

-- Promotion history
CREATE TABLE casino_data.promotion_history (
    promotion_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(20) REFERENCES casino_data.customer_demographics(customer_id),
    promotion_type VARCHAR(50),
    offer_date DATE,
    response BOOLEAN,
    response_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feature store for ML
CREATE TABLE casino_data.customer_features (
    feature_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(20) REFERENCES casino_data.customer_demographics(customer_id),
    feature_version VARCHAR(10),
    total_wagered DECIMAL(12,2),
    avg_bet_per_session DECIMAL(10,2),
    loss_rate DECIMAL(5,2),
    total_sessions INTEGER,
    days_since_last_visit INTEGER,
    avg_session_duration_min DECIMAL(10,2),
    bet_volatility DECIMAL(10,2),
    loss_chasing_score DECIMAL(5,4),
    multi_game_player BOOLEAN,
    weekend_preference DECIMAL(5,4),
    late_night_player DECIMAL(5,4),
    feature_created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(customer_id, feature_version)
);

-- Customer visit tracking (entry/exit)
CREATE TABLE casino_data.customer_visits (
    visit_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(20) REFERENCES casino_data.customer_demographics(customer_id),
    entry_time TIMESTAMP, -- with ±30min noise
    exit_time TIMESTAMP,
    visit_duration_minutes INTEGER,
    entry_point VARCHAR(50), -- 'Main', 'VIP', 'Parking'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Floor occupancy snapshots
CREATE TABLE casino_data.floor_occupancy (
    occupancy_id BIGSERIAL PRIMARY KEY,
    snapshot_time TIMESTAMP, -- hourly snapshots
    floor_zone VARCHAR(50),
    customer_count INTEGER,
    active_machines INTEGER,
    occupancy_rate DECIMAL(5,2), -- percentage
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- TITO (Ticket-In-Ticket-Out) transactions
CREATE TABLE casino_data.tito_transactions (
    transaction_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(20) REFERENCES casino_data.customer_demographics(customer_id),
    transaction_type VARCHAR(20), -- 'CASH_IN', 'CASH_OUT', 'TRANSFER'
    amount DECIMAL(10,2),
    machine_id VARCHAR(20),
    transaction_time TIMESTAMP, -- with noise
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Live dealer game sessions (different from slots)
CREATE TABLE casino_data.live_game_sessions (
    live_session_id BIGSERIAL PRIMARY KEY,
    customer_id VARCHAR(20) REFERENCES casino_data.customer_demographics(customer_id),
    game_type VARCHAR(50), -- 'Blackjack', 'Roulette', 'Poker'
    table_id VARCHAR(20),
    dealer_id VARCHAR(20), -- anonymized
    buy_in DECIMAL(10,2),
    cash_out DECIMAL(10,2),
    session_start TIMESTAMP,
    session_end TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- indexes
CREATE INDEX idx_visits_customer ON casino_data.customer_visits(customer_id);
CREATE INDEX idx_occupancy_time ON casino_data.floor_occupancy(snapshot_time);
CREATE INDEX idx_tito_customer ON casino_data.tito_transactions(customer_id);
CREATE INDEX idx_live_customer ON casino_data.live_game_sessions(customer_id);
-- Additional indexes
CREATE INDEX idx_visits_customer ON casino_data.customer_visits(customer_id);
CREATE INDEX idx_occupancy_time ON casino_data.floor_occupancy(snapshot_time);
CREATE INDEX idx_tito_customer ON casino_data.tito_transactions(customer_id);
CREATE INDEX idx_live_customer ON casino_data.live_game_sessions(customer_id);



-- Indexes for performance
CREATE INDEX idx_sessions_customer ON casino_data.player_sessions(customer_id);
CREATE INDEX idx_sessions_date ON casino_data.player_sessions(session_start);
CREATE INDEX idx_daily_customer_date ON casino_data.daily_aggregates(customer_id, date);
CREATE INDEX idx_features_customer ON casino_data.customer_features(customer_id);
CREATE INDEX idx_promotion_customer ON casino_data.promotion_history(customer_id);

-- Comments for documentation
COMMENT ON TABLE casino_data.customer_demographics IS 'Anonymized customer data - Ethics Ref: 10351-12382';
COMMENT ON TABLE casino_data.player_sessions IS 'Gaming sessions with temporal noise for privacy';
COMMENT ON TABLE casino_data.customer_features IS 'ML feature store - version controlled';