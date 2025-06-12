# src/features/feature_engineering.py

"""
Casino-Specific Feature Engineering
===================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

This module implements novel feature engineering specific to casino customer behavior.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

from ..config.academic_config import get_academic_header

logger = logging.getLogger(__name__)


class CasinoFeatureEngineer:
    """
    Original implementation of casino-specific feature engineering.
    
    Academic Note: This class represents a novel contribution to casino
    customer analytics by combining domain knowledge with ML best practices.
    """
    
    def __init__(self):
        self.feature_metadata = {
            "created_by": "Muhammed Yavuzhan CANLI",
            "methodology": "Domain-driven feature engineering",
            "academic_contribution": "Novel casino-specific features"
        }
        
    def create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create fundamental features from raw session data.
        
        Input columns expected:
        - customer_id
        - session_id
        - start_time
        - end_time
        - total_bet
        - total_win
        - game_type
        - machine_id
        """
        logger.info("Creating basic features from session data")
        
        features = pd.DataFrame()
        features['customer_id'] = df['customer_id'].unique()
        
        # Aggregate by customer
        customer_agg = df.groupby('customer_id').agg({
            'session_id': 'count',
            'total_bet': ['sum', 'mean', 'std'],
            'total_win': ['sum', 'mean'],
            'game_type': lambda x: x.mode().iloc[0] if len(x) > 0 else 'Unknown'
        }).reset_index()
        
        # Flatten column names
        customer_agg.columns = ['_'.join(col).strip() if col[1] else col[0] 
                               for col in customer_agg.columns.values]
        
        # Rename for clarity
        customer_agg.rename(columns={
            'session_id_count': 'total_sessions',
            'total_bet_sum': 'total_wagered',
            'total_bet_mean': 'avg_bet_per_session',
            'total_bet_std': 'bet_volatility',
            'total_win_sum': 'total_winnings',
            'total_win_mean': 'avg_win_per_session',
            'game_type_<lambda>': 'favorite_game_type'
        }, inplace=True)
        
        # Calculate derived features
        customer_agg['net_loss'] = customer_agg['total_wagered'] - customer_agg['total_winnings']
        customer_agg['loss_rate'] = (customer_agg['net_loss'] / 
                                     customer_agg['total_wagered'].replace(0, 1)) * 100
        customer_agg['avg_session_value'] = customer_agg['net_loss'] / customer_agg['total_sessions']
        
        # Risk indicators
        customer_agg['high_volatility_player'] = (
            customer_agg['bet_volatility'] > customer_agg['bet_volatility'].quantile(0.75)
        ).astype(int)
        
        return customer_agg
    
    def create_behavioral_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced behavioral features indicating player patterns.
        
        Academic contribution: Novel risk and engagement indicators.
        """
        features = pd.DataFrame()
        
        for customer_id in df['customer_id'].unique():
            customer_data = df[df['customer_id'] == customer_id].copy()
            customer_data = customer_data.sort_values('start_time')
            
            # Session patterns
            session_durations = (customer_data['end_time'] - customer_data['start_time']).dt.total_seconds() / 60
            
            # Time-based patterns
            customer_data['hour_of_day'] = customer_data['start_time'].dt.hour
            customer_data['day_of_week'] = customer_data['start_time'].dt.dayofweek
            
            behavioral_features = {
                'customer_id': customer_id,
                
                # Duration patterns
                'avg_session_duration_min': session_durations.mean(),
                'max_session_duration_min': session_durations.max(),
                'session_duration_volatility': session_durations.std(),
                
                # Time preferences
                'preferred_hour': customer_data['hour_of_day'].mode().iloc[0] if len(customer_data) > 0 else 0,
                'weekend_preference': (customer_data['day_of_week'] >= 5).mean(),
                'late_night_player': (customer_data['hour_of_day'] >= 22).mean(),
                
                # Engagement patterns
                'sessions_per_visit_day': customer_data.groupby(customer_data['start_time'].dt.date).size().mean(),
                'multi_game_player': (customer_data['game_type'].nunique() > 1).astype(int),
                
                # Risk indicators (original contribution)
                'rapid_play_indicator': (session_durations < 30).mean(),  # Short sessions
                'marathon_player': (session_durations > 180).any().astype(int),  # 3+ hour sessions
                'loss_chasing_score': self._calculate_loss_chasing_score(customer_data)
            }
            
            features = pd.concat([features, pd.DataFrame([behavioral_features])], ignore_index=True)
        
        return features
    
    def _calculate_loss_chasing_score(self, customer_data: pd.DataFrame) -> float:
        """
        Novel metric: Detect loss-chasing behavior patterns.
        
        Academic contribution: Original algorithm for identifying problematic gambling.
        Higher scores indicate potential loss-chasing behavior.
        """
        if len(customer_data) < 2:
            return 0.0
        
        customer_data['net_result'] = customer_data['total_win'] - customer_data['total_bet']
        customer_data['is_loss'] = (customer_data['net_result'] < 0).astype(int)
        
        # Check for increasing bets after losses
        loss_chasing_events = 0
        for i in range(1, len(customer_data)):
            if (customer_data.iloc[i-1]['is_loss'] == 1 and 
                customer_data.iloc[i]['total_bet'] > customer_data.iloc[i-1]['total_bet'] * 1.5):
                loss_chasing_events += 1
        
        return loss_chasing_events / len(customer_data)
    
    def create_feature_matrix(self, session_df: pd.DataFrame, 
                            include_behavioral: bool = True) -> pd.DataFrame:
        """
        Create complete feature matrix for ML models.
        
        Args:
            session_df: Raw session data
            include_behavioral: Whether to include complex behavioral features
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info(f"Creating feature matrix with behavioral={include_behavioral}")
        
        # Basic features
        basic_features = self.create_basic_features(session_df)
        
        if include_behavioral:
            # Behavioral features
            behavioral_features = self.create_behavioral_features(session_df)
            
            # Merge all features
            feature_matrix = basic_features.merge(
                behavioral_features, 
                on='customer_id', 
                how='left'
            )
        else:
            feature_matrix = basic_features
        
        # Add metadata
        feature_matrix['feature_version'] = '1.0'
        feature_matrix['created_at'] = datetime.now()
        
        logger.info(f"Created feature matrix with shape: {feature_matrix.shape}")
        return feature_matrix