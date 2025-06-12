# src/features/temporal_features.py

"""
Temporal Feature Engineering for Casino Analytics
================================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

Novel contribution: Time-series based features for casino customer behavior.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class TemporalFeatureEngineer:
    """
    Implements novel temporal feature engineering for casino analytics.
    
    Academic contribution: Combines time-series analysis with gambling behavior.
    """
    
    def __init__(self, time_windows: List[int] = [7, 14, 30, 90]):
        """
        Initialize with configurable time windows.
        
        Args:
            time_windows: List of days to look back for temporal features
        """
        self.time_windows = time_windows
        self.reference_date = None
        
    def set_reference_date(self, date: datetime):
        """Set the reference date for temporal calculations"""
        self.reference_date = date
        
    def create_recency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create recency-based features.
        
        Academic note: RFM (Recency, Frequency, Monetary) adaptation for casinos.
        """
        if self.reference_date is None:
            self.reference_date = df['start_time'].max()
            
        recency_features = []
        
        for customer_id in df['customer_id'].unique():
            customer_data = df[df['customer_id'] == customer_id]
            
            features = {
                'customer_id': customer_id,
                'days_since_last_visit': (self.reference_date - customer_data['start_time'].max()).days,
                'days_since_first_visit': (self.reference_date - customer_data['start_time'].min()).days,
                'customer_lifetime_days': (customer_data['start_time'].max() - customer_data['start_time'].min()).days
            }
            
            # Visit frequency patterns
            visit_dates = customer_data['start_time'].dt.date.unique()
            if len(visit_dates) > 1:
                visit_gaps = np.diff(sorted(visit_dates))
                features['avg_days_between_visits'] = np.mean([gap.days for gap in visit_gaps])
                features['visit_regularity_std'] = np.std([gap.days for gap in visit_gaps])
            else:
                features['avg_days_between_visits'] = 0
                features['visit_regularity_std'] = 0
                
            recency_features.append(features)
            
        return pd.DataFrame(recency_features)
    
    def create_trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create trend-based features showing behavior changes over time.
        
        Novel contribution: Detecting gambling pattern evolution.
        """
        trend_features = []
        
        for customer_id in df['customer_id'].unique():
            customer_data = df[df['customer_id'] == customer_id].sort_values('start_time')
            
            if len(customer_data) < 5:  # Need minimum sessions for trends
                continue
                
            # Split data into halves
            mid_point = len(customer_data) // 2
            first_half = customer_data.iloc[:mid_point]
            second_half = customer_data.iloc[mid_point:]
            
            features = {
                'customer_id': customer_id,
                
                # Betting trend
                'bet_trend_ratio': second_half['total_bet'].mean() / (first_half['total_bet'].mean() + 1),
                'session_frequency_trend': len(second_half) / (len(first_half) + 1),
                
                # Loss trend (risk indicator)
                'loss_rate_trend': (
                    ((second_half['total_bet'] - second_half['total_win']).sum() / second_half['total_bet'].sum()) /
                    ((first_half['total_bet'] - first_half['total_win']).sum() / first_half['total_bet'].sum() + 0.01)
                ),
                
                # Volatility trend
                'volatility_trend': second_half['total_bet'].std() / (first_half['total_bet'].std() + 1)
            }
            
            # Acceleration features (second derivative)
            if len(customer_data) >= 10:
                features['betting_acceleration'] = self._calculate_acceleration(
                    customer_data, 'total_bet'
                )
                
            trend_features.append(features)
            
        return pd.DataFrame(trend_features)
    
    def create_windowed_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features for different time windows.
        
        Academic contribution: Multi-scale temporal analysis.
        """
        if self.reference_date is None:
            self.reference_date = df['start_time'].max()
            
        windowed_features = []
        
        for customer_id in df['customer_id'].unique():
            customer_data = df[df['customer_id'] == customer_id]
            
            features = {'customer_id': customer_id}
            
            for window in self.time_windows:
                cutoff_date = self.reference_date - timedelta(days=window)
                window_data = customer_data[customer_data['start_time'] >= cutoff_date]
                
                if len(window_data) > 0:
                    features[f'sessions_last_{window}d'] = len(window_data)
                    features[f'total_bet_last_{window}d'] = window_data['total_bet'].sum()
                    features[f'avg_bet_last_{window}d'] = window_data['total_bet'].mean()
                    features[f'loss_rate_last_{window}d'] = (
                        (window_data['total_bet'] - window_data['total_win']).sum() / 
                        window_data['total_bet'].sum() * 100
                    )
                else:
                    features[f'sessions_last_{window}d'] = 0
                    features[f'total_bet_last_{window}d'] = 0
                    features[f'avg_bet_last_{window}d'] = 0
                    features[f'loss_rate_last_{window}d'] = 0
                    
            windowed_features.append(features)
            
        return pd.DataFrame(windowed_features)
    
    def _calculate_acceleration(self, data: pd.DataFrame, column: str) -> float:
        """Calculate acceleration (rate of change) for a metric"""
        if len(data) < 3:
            return 0.0
            
        # Simple numerical differentiation
        values = data[column].values
        first_derivative = np.diff(values)
        second_derivative = np.diff(first_derivative)
        
        return np.mean(second_derivative) if len(second_derivative) > 0 else 0.0
    
    def create_temporal_feature_matrix(self, session_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive temporal feature matrix.
        
        Returns:
            DataFrame with all temporal features
        """
        logger.info("Creating temporal feature matrix")
        
        # Create individual feature sets
        recency_features = self.create_recency_features(session_df)
        trend_features = self.create_trend_features(session_df)
        windowed_features = self.create_windowed_features(session_df)
        
        # Merge all temporal features
        temporal_matrix = recency_features
        
        if not trend_features.empty:
            temporal_matrix = temporal_matrix.merge(
                trend_features, on='customer_id', how='left'
            )
            
        temporal_matrix = temporal_matrix.merge(
            windowed_features, on='customer_id', how='left'
        )
        
        # Fill missing values
        temporal_matrix = temporal_matrix.fillna(0)
        
        logger.info(f"Created temporal feature matrix with shape: {temporal_matrix.shape}")
        return temporal_matrix