"""
Casino-Specific Feature Engineering
===================================
Compute customer features from valid game events and insert into PostgreSQL
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

This module implements novel feature engineering specific to casino customer behavior.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FixedCasinoFeatureEngine:
    """
    Database schema-compatible feature engineering for real casino data.
    
    Handles actual column names and missing columns gracefully.
    """
    
    def __init__(self):
        self.column_mapping = {
            # Map expected names to actual database columns
            'customer_id': 'player_id',  # Actual DB might use player_id
            'bet_col': 'bet_amount',     # Will auto-detect actual name
            'win_col': 'win_amount',     # Will auto-detect actual name
            'time_col': 'event_time',    # Time column
            'game_col': 'game_type',     # Game type column
            'machine_col': 'machine_id'  # Machine column
        }
    
    def detect_database_schema(self, df):
        """
        Auto-detect actual column names in the database.
        
        Returns mapping of standard names to actual column names.
        """
        logger.info("Auto-detecting database schema...")
        
        actual_columns = df.columns.tolist()
        detected_mapping = {}
        
        # Detect customer/player ID column
        id_candidates = ['player_id', 'customer_id', 'user_id', 'cust_id']
        detected_mapping['id_col'] = next((col for col in id_candidates if col in actual_columns), None)
        
        # Detect bet amount column
        bet_candidates = ['bet_amount', 'bet', 'amount', 'stake', 'wager']
        detected_mapping['bet_col'] = next((col for col in bet_candidates if col in actual_columns), None)
        
        # Detect win amount column  
        win_candidates = ['win_amount', 'win', 'payout', 'prize', 'winnings']
        detected_mapping['win_col'] = next((col for col in win_candidates if col in actual_columns), None)
        
        # Detect time column
        time_candidates = ['event_time', 'timestamp', 'time', 'datetime', 'ts']
        detected_mapping['time_col'] = next((col for col in time_candidates if col in actual_columns), None)
        
        # Detect game type column
        game_candidates = ['game_type', 'game', 'game_name', 'type']
        detected_mapping['game_col'] = next((col for col in game_candidates if col in actual_columns), None)
        
        # Detect machine column
        machine_candidates = ['machine_id', 'machine', 'terminal_id', 'device_id']
        detected_mapping['machine_col'] = next((col for col in machine_candidates if col in actual_columns), None)
        
        # Optional: zone/location column
        zone_candidates = ['zone_id', 'zone', 'area', 'location', 'floor']
        detected_mapping['zone_col'] = next((col for col in zone_candidates if col in actual_columns), None)
        
        # Log detected schema
        logger.info("Detected schema:")
        for key, value in detected_mapping.items():
            status = "✅ Found" if value else "❌ Missing"
            logger.info(f"  {key}: {value} ({status})")
        
        return detected_mapping
    
    def create_safe_feature_matrix(self, game_events_df):
        """
        Create feature matrix that handles missing columns gracefully.
        """
        logger.info(f"Processing {len(game_events_df):,} game events for feature engineering...")
        
        # Auto-detect schema
        schema = self.detect_database_schema(game_events_df)
        
        # Validate required columns
        if not schema['id_col']:
            raise ValueError("No customer/player ID column found!")
        if not schema['bet_col']:
            raise ValueError("No bet amount column found!")
        
        # Rename columns for consistency
        df = game_events_df.copy()
        df = df.rename(columns={
            schema['id_col']: 'customer_id',
            schema['bet_col']: 'bet_amount'
        })
        
        # Optional column renaming
        if schema['win_col']:
            df = df.rename(columns={schema['win_col']: 'win_amount'})
        else:
            df['win_amount'] = 0  # Default if missing
            
        if schema['time_col']:
            df = df.rename(columns={schema['time_col']: 'event_time'})
            
        if schema['game_col']:
            df = df.rename(columns={schema['game_col']: 'game_type'})
        else:
            df['game_type'] = 'Unknown'  # Default
            
        if schema['machine_col']:
            df = df.rename(columns={schema['machine_col']: 'machine_id'})
        else:
            df['machine_id'] = 'MACHINE_001'  # Default
        
        # Create features with safe column access
        features_df = self._compute_safe_features(df, schema)
        
        return features_df
    
    def _compute_safe_features(self, df, schema):
        """Compute features with safe column access."""
        logger.info("Computing features with safe column handling...")
        
        # Basic financial features
        basic_features = df.groupby('customer_id').agg({
            'bet_amount': ['count', 'sum', 'mean', 'std'],
            'win_amount': ['sum', 'mean']
        }).reset_index()
        
        # Flatten column names
        basic_features.columns = [
            'customer_id', 'total_events', 'total_bet', 'avg_bet', 'bet_std',
            'total_win', 'avg_win'
        ]
        
        # Calculate derived features
        basic_features['avg_loss'] = basic_features['total_bet'] - basic_features['total_win']
        basic_features['loss_rate'] = (
            basic_features['avg_loss'] / basic_features['total_bet']
        ).fillna(0).clip(0, 1)
        
        # Session-based features (using dates as session proxy)
        if 'event_time' in df.columns:
            df['event_date'] = pd.to_datetime(df['event_time']).dt.date
            session_features = df.groupby('customer_id')['event_date'].nunique().reset_index()
            session_features.columns = ['customer_id', 'total_sessions']
            basic_features = basic_features.merge(session_features, on='customer_id', how='left')
            
            # Session duration proxy
            basic_features['avg_events_per_session'] = (
                basic_features['total_events'] / basic_features['total_sessions']
            ).fillna(1)
        else:
            basic_features['total_sessions'] = basic_features['total_events'] // 10  # Estimate
            basic_features['avg_events_per_session'] = 10
        
        # Game diversity (if available)
        if 'game_type' in df.columns:
            game_diversity = df.groupby('customer_id')['game_type'].nunique().reset_index()
            game_diversity.columns = ['customer_id', 'game_diversity']
            basic_features = basic_features.merge(game_diversity, on='customer_id', how='left')
            basic_features['multi_game_player'] = (basic_features['game_diversity'] > 1).astype(int)
        else:
            basic_features['game_diversity'] = 1
            basic_features['multi_game_player'] = 0
        
        # Machine diversity (if available)
        if 'machine_id' in df.columns:
            machine_diversity = df.groupby('customer_id')['machine_id'].nunique().reset_index()
            machine_diversity.columns = ['customer_id', 'machine_diversity']
            basic_features = basic_features.merge(machine_diversity, on='customer_id', how='left')
        else:
            basic_features['machine_diversity'] = 1
        
        # Zone diversity (if available)
        if schema['zone_col'] and schema['zone_col'] in df.columns:
            zone_diversity = df.groupby('customer_id')[schema['zone_col']].nunique().reset_index()
            zone_diversity.columns = ['customer_id', 'zone_diversity']
            basic_features = basic_features.merge(zone_diversity, on='customer_id', how='left')
        else:
            basic_features['zone_diversity'] = 1
            logger.info("Zone column not available - using default zone_diversity = 1")
        
        # Risk indicators (safe computation)
        basic_features['bet_volatility'] = (
            basic_features['bet_std'] / basic_features['avg_bet']
        ).fillna(0).clip(0, 2)
        
        # Temporal features (if time column available)
        if 'event_time' in df.columns:
            temporal_features = self._compute_temporal_features(df)
            basic_features = basic_features.merge(temporal_features, on='customer_id', how='left')
        else:
            # Default temporal features
            basic_features['weekend_preference'] = 0.3  # Default
            basic_features['late_night_player'] = 0.1   # Default
            basic_features['days_since_last_visit'] = 30  # Default
            logger.info("Time column not available - using default temporal features")
        
        # Clean and validate features
        basic_features = self._clean_features(basic_features)
        
        logger.info(f"✅ Feature engineering completed for {len(basic_features):,} customers")
        
        return basic_features
    
    def _compute_temporal_features(self, df):
        """Compute temporal features if time column is available."""
        df['event_datetime'] = pd.to_datetime(df['event_time'])
        df['hour'] = df['event_datetime'].dt.hour
        df['dow'] = df['event_datetime'].dt.dayofweek
        
        temporal_features = df.groupby('customer_id').agg({
            'dow': lambda x: (x >= 5).mean(),  # Weekend preference
            'hour': lambda x: ((x >= 22) | (x <= 5)).mean(),  # Late night
            'event_datetime': ['min', 'max']
        }).reset_index()
        
        temporal_features.columns = [
            'customer_id', 'weekend_preference', 'late_night_player',
            'first_visit', 'last_visit'
        ]
        
        # Days since last visit
        current_date = datetime.now()
        temporal_features['days_since_last_visit'] = (
            current_date - temporal_features['last_visit']
        ).dt.days
        
        return temporal_features[['customer_id', 'weekend_preference', 'late_night_player', 'days_since_last_visit']]
    
    def _clean_features(self, features_df):
        """Clean and validate feature matrix."""
        initial_count = len(features_df)
        
        # Remove customers with insufficient data (less restrictive)
        features_df = features_df[features_df['total_events'] >= 5]  # Minimum 5 events
        features_df = features_df[features_df['total_bet'] > 0]
        
        # Handle outliers
        numeric_columns = features_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col != 'customer_id':
                upper_bound = features_df[col].quantile(0.99)
                features_df[col] = features_df[col].clip(upper=upper_bound)
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        final_count = len(features_df)
        logger.info(f"Feature validation: {initial_count} → {final_count} customers retained")
        
        return features_df

def main():
    """Main execution with error handling."""
    print("FIXED REAL CASINO FEATURE ENGINEERING")
    print("=" * 50)
    print("University of Bath - MSc Computer Science")
    print("Student: Muhammed Yavuzhan CANLI")
    print("Ethics Approval: 10351-12382")
    print("Database Schema Compatible Version")
    print("=" * 50)
    
    try:
        # Database connection - Update with your actual credentials
        engine = create_engine("postgresql://researcher:academic_password_2024@localhost:5432/casino_research")
        
        # Load game events
        logger.info("Loading game events data...")
        query = "SELECT * FROM casino_data.temp_valid_game_events LIMIT 100000"  # Test with limited data first
        game_events_df = pd.read_sql(query, engine)
        logger.info(f"Loaded {len(game_events_df):,} game events")
        
        # Show actual columns for verification
        logger.info(f"Actual columns in database: {list(game_events_df.columns)}")
        
        # Initialize feature engineer
        engineer = FixedCasinoFeatureEngine()
        
        # Create features with safe schema detection
        features_df = engineer.create_safe_feature_matrix(game_events_df)
        
        # Save to database
        logger.info("Saving features to database...")
        features_df.to_sql(
            "customer_features", 
            engine, 
            schema="casino_data", 
            if_exists="replace", 
            index=False
        )
        
        print(f"\n SUCCESS: Created {len(features_df):,} customer features")
        print("  Database schema compatibility ensured")
        print(" Ready for ML pipeline")
        print("\n  Next: python main_pipeline.py --mode=batch")
        print("=" * 50)
        
    except Exception as e:
        logger.error(f" !! - Feature engineering failed: {e}")
        print(" !! FAILED - Check database connection and schema")

if __name__ == "__main__":
    main()