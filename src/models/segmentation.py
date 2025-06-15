# src/models/segmentation.py

"""
Customer Segmentation using K-Means Clustering
==============================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

Implements casino-specific customer segmentation with academic rigor.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)


class CustomerSegmentation:
    """
    K-means based customer segmentation for casino analytics.
    
    Academic contribution: Domain-specific clustering with interpretable segments.
    """
    
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        """
        Initialize segmentation model.
        
        Args:
            n_clusters: Number of customer segments (default 4: casual, regular, high-roller, at-risk)
            random_state: For reproducibility (important for thesis)
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.segment_profiles = None
        
        # Academic metadata
        self.model_metadata = {
            "model_type": "KMeans_Clustering",
            "created_by": "Muhammed Yavuzhan CANLI",
            "academic_purpose": "Customer behavioral segmentation",
            "ethics_ref": "10351-12382"
        }
        
    def select_features(self, df: pd.DataFrame, 
                       feature_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Select and prepare features for clustering.
        
        Academic note: Feature selection based on domain knowledge and statistical relevance.
        """
        if feature_list is None:
            # Default features based on casino domain expertise
            feature_list = [
                'total_wagered',
                'avg_bet_per_session',
                'loss_rate',
                'total_sessions',
                'days_since_last_visit',
                'session_duration_volatility',
                'loss_chasing_score',
                'sessions_last_30d',
                'bet_trend_ratio'
            ]
        
        # Filter available features
        available_features = [f for f in feature_list if f in df.columns]
        self.feature_columns = available_features
        
        logger.info(f"Selected {len(available_features)} features for clustering")
        return df[available_features]
    
    def fit(self, df: pd.DataFrame, feature_list: Optional[List[str]] = None) -> 'CustomerSegmentation':
        """
        Fit the clustering model.
        
        Returns self for method chaining.
        """
        # Select features
        X = self.select_features(df, feature_list)
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit K-means
        logger.info(f"Fitting K-means with {self.n_clusters} clusters")
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        # Fit and predict
        labels = self.kmeans.fit_predict(X_scaled)
        
        # Calculate metrics
        silhouette = silhouette_score(X_scaled, labels)
        davies_bouldin = davies_bouldin_score(X_scaled, labels)
        
        logger.info(f"Clustering metrics - Silhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}")
        
        # Create segment profiles
        self._create_segment_profiles(df, X, labels)
        
        # Store metadata
        self.model_metadata.update({
            "fit_date": datetime.now().isoformat(),
            "n_samples": len(df),
            "n_features": len(self.feature_columns),
            "silhouette_score": silhouette,
            "davies_bouldin_score": davies_bouldin
        })
        
        return self
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict segment for new customers"""
        if self.kmeans is None:
            raise ValueError("Model must be fitted before prediction")
        
        X = df[self.feature_columns].fillna(df[self.feature_columns].median())
        X_scaled = self.scaler.transform(X)
        
        return self.kmeans.predict(X_scaled)
    
    def _create_segment_profiles(self, df: pd.DataFrame, X: pd.DataFrame, labels: np.ndarray):
        """
        Create interpretable profiles for each segment.
        
        Academic contribution: Business-meaningful segment interpretation.
        """
        df_profiling = df.copy()
        df_profiling['segment'] = labels
        
        # Calculate segment statistics
        profiles = {}
        
        for segment in range(self.n_clusters):
            segment_data = df_profiling[df_profiling['segment'] == segment]
            
            profile = {
                'segment_id': segment,
                'size': len(segment_data),
                'percentage': len(segment_data) / len(df) * 100,
                
                # Financial metrics
                'avg_total_wagered': segment_data['total_wagered'].mean(),
                'avg_loss_rate': segment_data['loss_rate'].mean(),
                'avg_session_value': segment_data.get('avg_session_value', segment_data['total_wagered']/segment_data['total_sessions']).mean(),
                
                # Behavioral metrics
                'avg_sessions': segment_data['total_sessions'].mean(),
                'avg_days_since_visit': segment_data['days_since_last_visit'].mean(),
                
                # Risk metrics
                'avg_loss_chasing_score': segment_data.get('loss_chasing_score', 0).mean(),
                'high_risk_percentage': (segment_data.get('loss_chasing_score', 0) > 0.3).mean() * 100
            }
            
            # Assign business label
            profile['business_label'] = self._assign_segment_label(profile)
            
            profiles[segment] = profile
        
        self.segment_profiles = profiles
        
    def _assign_segment_label(self, profile: Dict) -> str:
        """
        Assign meaningful business labels to segments.
        
        Academic note: Labels based on casino industry standards.
        """
        # Rule-based labeling (can be refined with domain experts)
        
        if profile['avg_total_wagered'] > 10000 and profile['avg_sessions'] > 20:
            return "High_Roller"
        elif profile['avg_loss_chasing_score'] > 0.3 or profile['high_risk_percentage'] > 40:
            return "At_Risk_Player"
        elif profile['avg_sessions'] > 10 and profile['avg_days_since_visit'] < 30:
            return "Regular_Visitor"
        else:
            return "Casual_Player"
    
    def get_segment_summary(self) -> pd.DataFrame:
        """Return segment profiles as DataFrame for analysis"""
        if self.segment_profiles is None:
            raise ValueError("Model must be fitted first")
        
        return pd.DataFrame.from_dict(self.segment_profiles, orient='index')
    
    def save_model(self, filepath: str):
        """Save model with academic metadata"""
        model_package = {
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'segment_profiles': self.segment_profiles,
            'metadata': self.model_metadata
        }
        
        joblib.dump(model_package, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load saved model"""
        model_package = joblib.load(filepath)
        
        self.kmeans = model_package['kmeans']
        self.scaler = model_package['scaler']
        self.feature_columns = model_package['feature_columns']
        self.segment_profiles = model_package['segment_profiles']
        self.model_metadata = model_package['metadata']
        
        logger.info(f"Model loaded from {filepath}")