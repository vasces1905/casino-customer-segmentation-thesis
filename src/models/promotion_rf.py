# src/models/promotion_rf.py

"""
Promotion Response Prediction using Random Forest
================================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

Predicts customer response to promotional offers using ensemble learning.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)


class PromotionResponseModel:
    """
    Random Forest model for predicting promotional offer response.
    
    Academic contribution: Multi-feature ensemble model with interpretability.
    """
    
    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        """
        Initialize Random Forest model.
        
        Args:
            n_estimators: Number of trees in the forest
            random_state: For reproducibility
        """
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.feature_importance = None
        
        # Academic metadata
        self.model_metadata = {
            "model_type": "RandomForestClassifier",
            "created_by": "Muhammed Yavuzhan CANLI",
            "academic_purpose": "Promotional response prediction",
            "ethics_ref": "10351-12382"
        }
        
    def prepare_features(self, df: pd.DataFrame, 
                        include_segment: bool = True) -> pd.DataFrame:
        """
        Prepare feature matrix for model training/prediction.
        
        Academic note: Combines behavioral, temporal, and segment features.
        """
        feature_list = [
            # Financial features
            'total_wagered',
            'avg_bet_per_session',
            'loss_rate',
            'net_loss',
            
            # Behavioral features
            'total_sessions',
            'avg_session_duration_min',
            'session_duration_volatility',
            'multi_game_player',
            'loss_chasing_score',
            
            # Temporal features
            'days_since_last_visit',
            'customer_lifetime_days',
            'sessions_last_30d',
            'bet_trend_ratio',
            'betting_acceleration',
            
            # Risk indicators
            'high_volatility_player',
            'marathon_player',
            'rapid_play_indicator'
        ]
        
        if include_segment and 'segment' in df.columns:
            feature_list.append('segment')
        
        # Filter available features
        available_features = [f for f in feature_list if f in df.columns]
        self.feature_columns = available_features
        
        logger.info(f"Using {len(available_features)} features for promotion model")
        return df[available_features]
    
    def fit(self, X: pd.DataFrame, y: np.ndarray, 
            validation_split: float = 0.2) -> 'PromotionResponseModel':
        """
        Train the Random Forest model.
        
        Args:
            X: Feature matrix
            y: Target labels (0: no response, 1: positive response)
            validation_split: Fraction for validation
            
        Returns:
            Self for method chaining
        """
        # Prepare features
        X_prep = self.prepare_features(X)
        X_prep = X_prep.fillna(X_prep.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_prep)
        
        # Initialize model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            max_depth=10,  # Prevent overfitting
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced'  # Handle imbalanced classes
        )
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, X_scaled, y, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state),
            scoring='roc_auc'
        )
        
        logger.info(f"Cross-validation ROC-AUC: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Fit model
        self.model.fit(X_scaled, y)
        
        # Calculate feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store metadata
        self.model_metadata.update({
            "fit_date": datetime.now().isoformat(),
            "n_samples": len(X),
            "n_features": len(self.feature_columns),
            "cv_roc_auc": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "class_distribution": pd.Series(y).value_counts().to_dict()
        })
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict promotion response"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_prep = X[self.feature_columns].fillna(X[self.feature_columns].median())
        X_scaled = self.scaler.transform(X_prep)
        
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict promotion response probability"""
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_prep = X[self.feature_columns].fillna(X[self.feature_columns].median())
        X_scaled = self.scaler.transform(X_prep)
        
        return self.model.predict_proba(X_scaled)
    
    def evaluate(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """
        Comprehensive model evaluation.
        
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        # Classification report
        report = classification_report(y, y_pred, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # ROC-AUC
        roc_auc = roc_auc_score(y, y_proba)
        
        evaluation_results = {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'accuracy': report['accuracy'],
            'precision_positive': report['1']['precision'],
            'recall_positive': report['1']['recall'],
            'f1_positive': report['1']['f1-score']
        }
        
        logger.info(f"Model evaluation - ROC-AUC: {roc_auc:.3f}, Accuracy: {report['accuracy']:.3f}")
        
        return evaluation_results
    
    def get_feature_importance(self, top_n: int = 10) -> pd.DataFrame:
        """Get top N most important features"""
        if self.feature_importance is None:
            raise ValueError("Model must be fitted first")
        
        return self.feature_importance.head(top_n)
    
    def save_model(self, filepath: str):
        """Save model with metadata"""
        model_package = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'feature_importance': self.feature_importance,
            'metadata': self.model_metadata
        }
        
        joblib.dump(model_package, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load saved model"""
        model_package = joblib.load(filepath)
        
        self.model = model_package['model']
        self.scaler = model_package['scaler']
        self.feature_columns = model_package['feature_columns']
        self.feature_importance = model_package['feature_importance']
        self.model_metadata = model_package['metadata']
        
        logger.info(f"Model loaded from {filepath}")