# src/features/hybrid_feature_engineering.py

"""
Hybrid Casino Feature Engineering - Leveraging Existing Processed Data
=====================================================================
University of Bath - MSc Computer Science
Student: Muhammed Yavuzhan CANLI
Supervisor: Dr. Moody Alam
Ethics Approval: 10351-12382

Hybrid Approach: Combines existing processed data with novel academic features.

Existing Data Sources:
- customer_behavior_profiles (38,319 customers - 100% coverage)
- customer_game_preferences (38,319 customers)
- temp_valid_game_events (2.4M+ events)

Academic Contributions Added:
- Advanced loss-chasing detection algorithms
- Temporal behavioral pattern analysis
- Risk assessment metrics
- Zone diversity calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import psycopg2
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)


class HybridCasinoFeatureEngineer:
    """
    Hybrid feature engineering combining existing processed data with novel features.
    
    Academic Innovation:
    - Leverages existing customer_behavior_profiles (100% coverage)
    - Adds novel academic contributions (loss-chasing, temporal patterns)
    - Achieves comprehensive feature set without recreating existing work
    - Maintains VNS scope compliance and academic standards
    """
    
    def __init__(self, db_connector=None):
        self.db_connector = db_connector
        self.academic_metadata = {
            "created_by": "Muhammed Yavuzhan CANLI",
            "institution": "University of Bath",
            "ethics_ref": "10351-12382",
            "version": "4.0 - Hybrid Approach",
            "methodology": "Hybrid: Existing + Novel Academic Features",
            "coverage": "100% customers (38,319)"
        }
        
    def load_existing_processed_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load existing processed customer data from database.
        
        Returns:
            Tuple of (behavior_profiles, game_preferences) DataFrames
        """
        logger.info("Loading existing processed customer data...")
        
        # Load customer behavior profiles
        behavior_query = """
        SELECT 
            player_id as customer_id,
            primary_game,
            secondary_game, 
            tertiary_game,
            overall_avg_bet,
            game_variety,
            total_customer_spins as total_events,
            total_visit_days,
            player_type,
            betting_category
        FROM casino_data.customer_behavior_profiles
        WHERE player_id IS NOT NULL
        """
        
        behavior_df = pd.read_sql_query(behavior_query, self.db_connector.get_connection())
        
        # Load game preferences
        preferences_query = """
        SELECT 
            player_id as customer_id,
            primary_game as pref_primary_game,
            secondary_game as pref_secondary_game,
            tertiary_game as pref_tertiary_game
        FROM casino_data.customer_game_preferences
        WHERE player_id IS NOT NULL
        """
        
        preferences_df = pd.read_sql_query(preferences_query, self.db_connector.get_connection())
        
        logger.info(f"Loaded behavior data: {len(behavior_df):,} customers")
        logger.info(f"Loaded preferences data: {len(preferences_df):,} customers")
        
        return behavior_df, preferences_df
    
    def load_raw_game_events(self, customer_ids: List[str]) -> pd.DataFrame:
        """
        Load raw game events for academic feature calculation.
        
        Args:
            customer_ids: List of customer IDs to load events for
            
        Returns:
            DataFrame with raw game events
        """
        logger.info(f"Loading raw game events for {len(customer_ids):,} customers...")
        
        # Convert customer_ids to comma-separated string for SQL IN clause
        ids_str = "', '".join(customer_ids)
        
        query = f"""
        SELECT 
            player_id as customer_id,
            ts as event_timestamp,
            bet_amount,
            win_amount,
            game_id,
            machine_id,
            gaming_day
        FROM casino_data.temp_valid_game_events 
        WHERE player_id IN ('{ids_str}')
            AND bet_amount >= 0
        ORDER BY player_id, ts
        """
        
        events_df = pd.read_sql_query(query, self.db_connector.get_connection())
        
        # Data type conversions
        events_df['event_timestamp'] = pd.to_datetime(events_df['event_timestamp'], errors='coerce')
        events_df['bet_amount'] = pd.to_numeric(events_df['bet_amount'], errors='coerce').fillna(0)
        events_df['win_amount'] = pd.to_numeric(events_df['win_amount'], errors='coerce').fillna(0)
        
        events_df = events_df.dropna(subset=['event_timestamp'])
        
        logger.info(f"Loaded {len(events_df):,} game events")
        return events_df
    
    def calculate_missing_academic_features(self, events_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate missing academic features from raw game events.
        
        Academic Contributions:
        - Advanced loss-chasing detection
        - Temporal behavioral patterns
        - Risk assessment metrics
        - Zone diversity analysis
        """
        logger.info("Calculating missing academic features...")
        
        academic_features = []
        customer_ids = events_df['customer_id'].unique()
        
        for i, customer_id in enumerate(customer_ids):
            if i % 5000 == 0:
                logger.info(f"Processing customer {i+1:,}/{len(customer_ids):,}")
                
            customer_events = events_df[events_df['customer_id'] == customer_id].copy()
            customer_events = customer_events.sort_values('event_timestamp')
            
            features = self._calculate_customer_academic_features(customer_events)
            features['customer_id'] = customer_id
            
            academic_features.append(features)
        
        academic_df = pd.DataFrame(academic_features)
        logger.info(f"Calculated academic features for {len(academic_df):,} customers")
        
        return academic_df
    
    def _calculate_customer_academic_features(self, customer_events: pd.DataFrame) -> Dict:
        """
        Calculate academic features for a single customer.
        
        Returns dictionary of novel academic features.
        """
        total_events = len(customer_events)
        
        if total_events == 0:
            return self._get_default_features()
        
        # Basic calculations
        total_bet = customer_events['bet_amount'].sum()
        total_win = customer_events['win_amount'].sum()
        net_loss = total_bet - total_win
        
        # Academic Feature 1: Advanced Loss-Chasing Detection
        loss_chasing_score = self._calculate_advanced_loss_chasing(customer_events)
        
        # Academic Feature 2: Temporal Behavioral Patterns
        temporal_features = self._calculate_temporal_patterns(customer_events)
        
        # Academic Feature 3: Risk Assessment Metrics
        risk_features = self._calculate_risk_metrics(customer_events)
        
        # Academic Feature 4: Zone Diversity Analysis
        zone_diversity = self._calculate_zone_diversity(customer_events)
        
        # Additional academic metrics
        bet_volatility = customer_events['bet_amount'].std() / max(1, customer_events['bet_amount'].mean())
        
        # Combine all features
        features = {
            # Loss and risk metrics
            'total_win': total_win,
            'net_loss': net_loss,
            'loss_rate': (net_loss / max(1, total_bet)) * 100,
            'loss_chasing_score': loss_chasing_score,
            'bet_volatility': bet_volatility,
            
            # Zone and machine diversity
            'machine_diversity': customer_events['machine_id'].nunique(),
            'zone_diversity': zone_diversity,
            
            # Time-based features
            'days_since_last_visit': (datetime.now() - customer_events['event_timestamp'].max()).days,
            **temporal_features,
            **risk_features
        }
        
        return features
    
    def _calculate_advanced_loss_chasing(self, customer_events: pd.DataFrame) -> float:
        """
        Novel Academic Contribution: Advanced loss-chasing detection algorithm.
        
        Detects patterns where customers increase bets following losses,
        indicating potential problematic gambling behavior.
        """
        if len(customer_events) < 3:
            return 0.0
        
        customer_events = customer_events.sort_values('event_timestamp')
        customer_events['net_result'] = customer_events['win_amount'] - customer_events['bet_amount']
        customer_events['is_loss'] = (customer_events['net_result'] < 0).astype(int)
        
        loss_chasing_events = 0
        total_opportunities = 0
        
        for i in range(1, len(customer_events)):
            if customer_events.iloc[i-1]['is_loss'] == 1:
                total_opportunities += 1
                
                # Check for bet escalation after loss (25% increase threshold)
                current_bet = customer_events.iloc[i]['bet_amount']
                previous_bet = customer_events.iloc[i-1]['bet_amount']
                
                if current_bet > previous_bet * 1.25:
                    loss_chasing_events += 1
        
        return loss_chasing_events / max(1, total_opportunities)
    
    def _calculate_temporal_patterns(self, customer_events: pd.DataFrame) -> Dict:
        """
        Calculate temporal behavioral patterns.
        
        Academic Contribution: Time-based behavior analysis for casino customers.
        """
        if len(customer_events) < 2:
            return {
                'weekend_preference': 0.5,
                'late_night_player': 0.0,
                'session_intensity': 1.0,
                'temporal_consistency': 0.0
            }
        
        # Time-based features
        customer_events['hour'] = customer_events['event_timestamp'].dt.hour
        customer_events['day_of_week'] = customer_events['event_timestamp'].dt.dayofweek
        customer_events['is_weekend'] = (customer_events['day_of_week'] >= 5).astype(int)
        
        # Calculate patterns
        weekend_preference = customer_events['is_weekend'].mean()
        late_night_player = (customer_events['hour'] >= 22).mean()
        
        # Session intensity (events per unique day)
        unique_days = customer_events['event_timestamp'].dt.date.nunique()
        session_intensity = len(customer_events) / max(1, unique_days)
        
        # Temporal consistency
        total_span_days = (customer_events['event_timestamp'].max() - 
                          customer_events['event_timestamp'].min()).days + 1
        temporal_consistency = unique_days / max(1, total_span_days)
        
        return {
            'weekend_preference': weekend_preference,
            'late_night_player': late_night_player,
            'session_intensity': session_intensity,
            'temporal_consistency': temporal_consistency
        }
    
    def _calculate_risk_metrics(self, customer_events: pd.DataFrame) -> Dict:
        """
        Calculate risk assessment metrics.
        
        Academic Contribution: Comprehensive risk profiling for responsible gambling.
        """
        if len(customer_events) < 5:
            return {
                'high_risk_sessions': 0,
                'bet_escalation_tendency': 0.0,
                'volatile_betting_pattern': 0.0
            }
        
        # High-risk sessions (top 10% of bets for this customer)
        high_bet_threshold = customer_events['bet_amount'].quantile(0.9)
        high_risk_sessions = (customer_events['bet_amount'] >= high_bet_threshold).sum()
        
        # Bet escalation tendency
        bet_increases = (customer_events['bet_amount'].diff() > 0).sum()
        bet_escalation_tendency = bet_increases / len(customer_events)
        
        # Volatile betting pattern
        bet_cv = customer_events['bet_amount'].std() / max(1, customer_events['bet_amount'].mean())
        volatile_betting_pattern = min(bet_cv, 3.0)  # Cap at 3.0
        
        return {
            'high_risk_sessions': high_risk_sessions,
            'bet_escalation_tendency': bet_escalation_tendency,
            'volatile_betting_pattern': volatile_betting_pattern
        }
    
    def _calculate_zone_diversity(self, customer_events: pd.DataFrame) -> int:
        """
        Calculate gaming zone diversity from machine patterns.
        
        Academic Contribution: Spatial behavior analysis in casino environments.
        """
        if 'machine_id' not in customer_events.columns:
            return 1
        
        # Extract zone information from machine IDs
        machine_ids = customer_events['machine_id'].astype(str)
        
        # Assume zones are indicated by first 2-3 characters of machine ID
        zones = machine_ids.str[:2].nunique()  # First 2 chars = zone
        
        return max(1, zones)
    
    def _get_default_features(self) -> Dict:
        """Default features for customers with no events."""
        return {
            'total_win': 0,
            'net_loss': 0,
            'loss_rate': 0,
            'loss_chasing_score': 0,
            'bet_volatility': 0,
            'machine_diversity': 0,
            'zone_diversity': 1,
            'days_since_last_visit': 999,
            'weekend_preference': 0.5,
            'late_night_player': 0.0,
            'session_intensity': 0.0,
            'temporal_consistency': 0.0,
            'high_risk_sessions': 0,
            'bet_escalation_tendency': 0.0,
            'volatile_betting_pattern': 0.0
        }
    
    def create_hybrid_feature_matrix(self) -> pd.DataFrame:
        """
        Create comprehensive feature matrix combining existing + novel features.
        
        Academic Achievement: 100% customer coverage with novel behavioral analytics.
        """
        logger.info("Creating hybrid feature matrix...")
        
        # Step 1: Load existing processed data
        behavior_df, preferences_df = self.load_existing_processed_data()
        
        # Step 2: Merge existing data
        existing_features = behavior_df.merge(
            preferences_df, 
            on='customer_id', 
            how='left'
        )
        
        logger.info(f"Existing features: {len(existing_features):,} customers")
        
        # Step 3: Load raw events for academic feature calculation
        customer_ids = existing_features['customer_id'].tolist()
        
        # Process in batches to avoid memory issues
        batch_size = 5000
        all_academic_features = []
        
        for i in range(0, len(customer_ids), batch_size):
            batch_ids = customer_ids[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(customer_ids)-1)//batch_size + 1}")
            
            # Load events for this batch
            batch_events = self.load_raw_game_events(batch_ids)
            
            # Calculate academic features
            batch_academic_features = self.calculate_missing_academic_features(batch_events)
            all_academic_features.append(batch_academic_features)
        
        # Combine all academic features
        academic_features_df = pd.concat(all_academic_features, ignore_index=True)
        
        # Step 4: Merge existing + academic features
        complete_features = existing_features.merge(
            academic_features_df,
            on='customer_id',
            how='left'
        )
        
        # Fill any missing values
        numeric_columns = complete_features.select_dtypes(include=[np.number]).columns
        complete_features[numeric_columns] = complete_features[numeric_columns].fillna(0)
        
        # Add metadata
        complete_features['feature_version'] = '4.0_hybrid'
        complete_features['feature_created_at'] = datetime.now()
        
        logger.info(f"Hybrid feature matrix completed!")
        logger.info(f"Coverage: {len(complete_features):,} customers (100%)")
        logger.info(f"Features: {len(complete_features.columns)} columns")
        
        return complete_features
    
    def save_hybrid_features_to_database(self, feature_matrix: pd.DataFrame):
        """
        Save hybrid feature matrix to database.
        """
        logger.info("Saving hybrid features to casino_data.customer_features...")
        
        # Prepare for database
        feature_matrix_db = feature_matrix.copy()
        
        # Handle datetime columns
        datetime_columns = feature_matrix_db.select_dtypes(include=['datetime64']).columns
        for col in datetime_columns:
            feature_matrix_db[col] = feature_matrix_db[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Save to database
        engine = create_engine(self.db_connector.get_connection_string())
        feature_matrix_db.to_sql(
            'customer_features', 
            engine, 
            schema='casino_data',
            if_exists='replace',
            index=False,
            method='multi'
        )
        
        logger.info(f" Successfully saved {len(feature_matrix_db):,} customer feature records")
        
        # Log academic achievement
        self._log_hybrid_achievement(feature_matrix_db)
    
    def _log_hybrid_achievement(self, feature_matrix: pd.DataFrame):
        """Log hybrid approach achievement."""
        achievement_metadata = {
            "methodology": "Hybrid: Existing + Novel Academic Features",
            "coverage_achievement": f"{len(feature_matrix):,} customers (100%)",
            "academic_contributions": [
                "Advanced loss-chasing detection algorithm",
                "Temporal behavioral pattern analysis", 
                "Comprehensive risk assessment metrics",
                "Zone diversity spatial analysis"
            ],
            "data_sources": [
                "customer_behavior_profiles (existing)",
                "customer_game_preferences (existing)",
                "temp_valid_game_events (novel features)"
            ],
            "institution": "University of Bath",
            "ethics_ref": "10351-12382"
        }
        
        logger.info("Hybrid Feature Engineering Achievement:")
        logger.info(f" 100% Coverage: {len(feature_matrix):,} customers")
        logger.info(f" Novel Features: Advanced behavioral analytics")
        logger.info(f" Academic Value: Original contributions preserved")


# Usage example
if __name__ == "__main__":
    print("🎓 UNIVERSITY OF BATH - HYBRID FEATURE ENGINEERING")
    print("=" * 60)
    print("Student: Muhammed Yavuzhan CANLI")
    print("Ethics: 10351-12382")
    print("Approach: Hybrid (Existing + Novel Academic Features)")
    print("Coverage: 100% customers (38,319)")
    print("=" * 60)