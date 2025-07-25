# robust_rf_training_major_fix.py - RF Training After Aggressive Outlier Control
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RobustRFTrainer:
    """RF Training with major fix applied"""
    
    def __init__(self):
        self.scaler = RobustScaler(quantile_range=(10.0, 90.0))
        self.label_encoder = LabelEncoder()
        self.rf_model = None
        self.feature_names = None
        self.cv_results = {}
        
    def prepare_robust_features(self, df):
        """Select and prepare robust features after major fix"""
        logger.info("🔧 PREPARING ROBUST FEATURES")
        
        # ROBUST FEATURE SET (outlier-resistant)
        robust_features = [
            # Period-normalized features (major fix additions)
            'bet_percentile_in_period',
            'risk_percentile_in_period', 
            'session_percentile_in_period',
            'bet_zscore_in_period',
            'robust_risk_score',
            
            # Capped/transformed features  
            'log_bet_stable',  # Log of winsorized bet
            'total_sessions',  # Stable across periods
            'loss_rate',       # Stable metric
            
            # Segment features
            'kmeans_segment_encoded',
            'is_dbscan_outlier',
            'is_stable_customer',
            
            # Original features (for comparison)
            'loss_chasing_score',
            'session_duration_volatility'
        ]
        
        # Ensure all features exist
        available_features = [f for f in robust_features if f in df.columns]
        missing_features = [f for f in robust_features if f not in df.columns]
        
        if missing_features:
            logger.warning(f"   Missing features: {missing_features}")
        
        logger.info(f"   Using {len(available_features)} robust features")
        self.feature_names = available_features
        
        # Prepare feature matrix
        X = df[available_features].fillna(0)
        
        # Handle any remaining infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return X
    
    def train_robust_model(self, df):
        """Train RF with robust features and enhanced sample weights"""
        logger.info("🚀 TRAINING ROBUST RF MODEL")
        
        # Prepare features
        X = self.prepare_robust_features(df)
        y = df['robust_promotion_label']
        sample_weights = df['enhanced_sample_weight']
        groups = df['analysis_period']  # For group-aware CV
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Enhanced RF model (tuned for outlier resistance)
        self.rf_model = RandomForestClassifier(
            n_estimators=300,     # More trees for stability
            max_depth=8,          # Reduced depth to prevent overfitting
            min_samples_split=20, # Higher to prevent outlier dominance
            min_samples_leaf=10,  # Higher for stability
            max_features='sqrt',  # Feature randomness
            bootstrap=True,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Group-stratified cross-validation (prevent data leakage across periods)
        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        
        # Cross-validation with sample weights
        logger.info("   Running stratified group cross-validation...")
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(cv.split(X_scaled, y_encoded, groups)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]
            w_train = sample_weights.iloc[train_idx]
            
            # Train fold model
            fold_model = RandomForestClassifier(**self.rf_model.get_params())
            fold_model.fit(X_train, y_train, sample_weight=w_train)
            
            # Evaluate
            fold_score = fold_model.score(X_val, y_val)
            cv_scores.append(fold_score)
            
            logger.info(f"   Fold {fold+1}: {fold_score:.4f}")
        
        # Train final model on full dataset
        logger.info("   Training final model on full dataset...")
        self.rf_model.fit(X_scaled, y_encoded, sample_weight=sample_weights)
        
        # Store CV results
        self.cv_results = {
            'scores': cv_scores,
            'mean': np.mean(cv_scores),
            'std': np.std(cv_scores),
            'min': np.min(cv_scores),
            'max': np.max(cv_scores)
        }
        
        # Feature importance analysis
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.rf_model.feature_importances_,
            'category': self._categorize_features()
        }).sort_values('importance', ascending=False)
        
        # Results summary
        logger.info(f"✅ ROBUST RF TRAINING COMPLETED!")
        logger.info(f"   CV Mean Accuracy: {self.cv_results['mean']:.4f} (±{self.cv_results['std']:.4f})")
        logger.info(f"   CV Range: {self.cv_results['min']:.4f} - {self.cv_results['max']:.4f}")
        
        print("\n🎯 TOP 10 FEATURE IMPORTANCE:")
        print(feature_importance_df.head(10).to_string(index=False))
        
        # Category importance
        category_importance = feature_importance_df.groupby('category')['importance'].sum().sort_values(ascending=False)
        print(f"\n📊 FEATURE CATEGORY IMPORTANCE:")
        for category, importance in category_importance.items():
            print(f"   {category}: {importance:.3f} ({importance/category_importance.sum()*100:.1f}%)")
        
        return {
            'cv_results': self.cv_results,
            'feature_importance': feature_importance_df,
            'category_importance': category_importance,
            'model_performance': 'robust'
        }
    
    def _categorize_features(self):
        """Categorize features for analysis"""
        categories = []
        for feature in self.feature_names:
            if 'percentile_in_period' in feature or 'zscore_in_period' in feature:
                categories.append('period_normalized')
            elif feature in ['robust_risk_score', 'is_stable_customer']:
                categories.append('robust_engineered')
            elif feature in ['kmeans_segment_encoded', 'is_dbscan_outlier']:
                categories.append('segmentation')
            elif feature in ['log_bet_stable']:
                categories.append('transformed_financial')
            else:
                categories.append('original_behavioral')
        return categories
    
    def validate_against_periods(self, df):
        """Validate model performance across individual periods"""
        logger.info("🔍 VALIDATING MODEL ACROSS PERIODS")
        
        X = self.prepare_robust_features(df)
        X_scaled = self.scaler.transform(X)
        y_true = self.label_encoder.transform(df['robust_promotion_label'])
        
        period_results = {}
        
        for period in df['analysis_period'].unique():
            period_mask = df['analysis_period'] == period
            X_period = X_scaled[period_mask]
            y_period = y_true[period_mask]
            
            if len(np.unique(y_period)) > 1:  # Need multiple classes for accuracy
                period_score = self.rf_model.score(X_period, y_period)
                period_results[period] = period_score
                logger.info(f"   {period}: {period_score:.4f}")
            else:
                period_results[period] = "single_class"
                logger.info(f"   {period}: single class only")
        
        return period_results
    
    def save_robust_model(self, output_dir='models/robust_major_fix'):
        """Save the robust model with major fix"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = f"{output_dir}/robust_rf_major_fix_{datetime.now().strftime('%Y%m%d_%H%M')}.pkl"
        
        model_package = {
            'rf_model': self.rf_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'cv_results': self.cv_results,
            'preprocessing': 'major_fix_aggressive_outlier_control',
            'model_type': 'robust_rf_with_period_normalization',
            'training_date': datetime.now().isoformat(),
            'outlier_handling': 'mega_outlier_removal_99.5_percentile_capping',
            'sample_weighting': 'enhanced_with_outlier_penalties'
        }
        
        joblib.dump(model_package, model_path)
        logger.info(f"✅ Robust model saved to: {model_path}")
        
        return model_path

def main():
    """Main training pipeline with major fix"""
    logger.info("🚀 STARTING ROBUST RF TRAINING - MAJOR FIX VERSION")
    
    # This assumes aggressive_outlier_control.py has been run first
    # and df_processed is available
    
    print("📋 PREREQUISITES:")
    print("   1. Run: python aggressive_outlier_control.py")
    print("   2. Ensure df_processed is saved/available")
    print("   3. Then run this robust RF training")
    print("\n🎯 EXPECTED MAJOR FIX RESULTS:")
    print("   • CV accuracy: >0.85 (stable across folds)")
    print("   • Max bet values: <€10M (controlled)")
    print("   • Feature importance: Period-normalized features dominant")
    print("   • Model stability: Consistent across all periods")
    
    logger.info("✅ MAJOR FIX ROBUST RF TRAINER READY!")

if __name__ == "__main__":
    main()