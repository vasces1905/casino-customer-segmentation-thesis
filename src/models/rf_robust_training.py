# rf_robust_training.py - Anomaly-Aware RF Model
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
import warnings
warnings.filterwarnings('ignore')

class RobustRFModel:
    """RF Model with 2023-H2 anomaly handling"""
    
    def __init__(self):
        self.scaler = RobustScaler(quantile_range=(25.0, 75.0))  # Robust to outliers
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_split=10,
            random_state=42,
            class_weight='balanced'
        )
    
    def preprocess_anomaly_data(self, df):
        """Handle 2023-H2 anomalies"""
        df_clean = df.copy()
        
        # 1. CAP EXTREME OUTLIER VALUES (99th percentile)
        outlier_bet_cap = df['total_bet'].quantile(0.99)
        df_clean['total_bet_capped'] = df['total_bet'].clip(upper=outlier_bet_cap)
        
        # 2. LOG TRANSFORM EXTREME VALUES
        df_clean['log_total_bet'] = np.log1p(df['total_bet'])
        
        # 3. PERIOD-AWARE SAMPLE WEIGHTS
        period_weights = {
            '2022-H1': 1.0,
            '2022-H2': 1.0,
            '2023-H1': 1.0,
            '2023-H2': 0.4  # Reduce anomaly period influence
        }
        df_clean['sample_weight'] = df_clean['analysis_period'].map(period_weights)
        
        return df_clean
    
    def create_robust_features(self, df):
        """Create stable features across all periods"""
        df_robust = df.copy()
        
        # ROBUST RISK FEATURES (less sensitive to extremes)
        df_robust['risk_percentile'] = df.groupby('analysis_period')['loss_chasing_score'].rank(pct=True)
        df_robust['bet_percentile'] = df.groupby('analysis_period')['total_bet'].rank(pct=True)
        
        # PERIOD-NORMALIZED FEATURES
        df_robust['bet_vs_period_median'] = df.groupby('analysis_period')['total_bet'].transform(
            lambda x: x / x.median()
        )
        
        return df_robust
    
    def train_with_anomaly_handling(self, df):
        """Train RF with robust anomaly handling"""
        
        # Preprocess data
        df_processed = self.preprocess_anomaly_data(df)
        df_robust = self.create_robust_features(df_processed)
        
        # Feature selection (robust features only)
        robust_features = [
            'log_total_bet', 'bet_percentile', 'risk_percentile',
            'total_sessions', 'loss_rate', 'bet_vs_period_median',
            'is_dbscan_outlier', 'kmeans_segment_encoded'
        ]
        
        # Prepare training data
        X = df_robust[robust_features].fillna(0)
        y = df_robust['promotion_label']
        sample_weights = df_robust['sample_weight']
        groups = df_robust['analysis_period']  # For group-aware CV
        
        # Robust scaling
        X_scaled = self.scaler.fit_transform(X)
        
        # Stratified Group K-Fold (prevents data leakage across periods)
        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        
        # Train model with sample weights
        self.rf_model.fit(X_scaled, y, sample_weight=sample_weights)
        
        # Validation across periods
        cv_scores = []
        for train_idx, val_idx in cv.split(X_scaled, y, groups):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            w_train = sample_weights.iloc[train_idx]
            
            temp_model = RandomForestClassifier(**self.rf_model.get_params())
            temp_model.fit(X_train, y_train, sample_weight=w_train)
            score = temp_model.score(X_val, y_val)
            cv_scores.append(score)
        
        print(f"✅ Robust RF Training Complete!")
        print(f"   Cross-validation scores: {cv_scores}")
        print(f"   Mean CV accuracy: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
        
        return {
            'cv_scores': cv_scores,
            'mean_accuracy': np.mean(cv_scores),
            'feature_names': robust_features,
            'anomaly_handling': 'enabled'
        }

# Usage example:
# model = RobustRFModel()
# results = model.train_with_anomaly_handling(df_combined)