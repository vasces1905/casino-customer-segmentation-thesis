# optimized_kmeans_dbscan_rf.py - Streamlined K-Means + DBSCAN Approach
import numpy as np
import pandas as pd
import psycopg2
import json
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedRFPromotionModel:
    """Streamlined RF model using only K-Means + DBSCAN for optimal business value"""
    
    def __init__(self, period_id: str):
        self.period_id = period_id
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.rf_model = None
        self.feature_names = None
        self.feature_importance_df = None
        
        # Simplified feature categories - Focus on business value
        self.base_behavioral_features = [
            'total_bet', 'avg_bet', 'loss_rate', 'total_sessions',
            'days_since_last_visit', 'session_duration_volatility', 
            'loss_chasing_score', 'sessions_last_30d', 'bet_trend_ratio'
        ]
        
        # Core enhancement features - Only K-Means + DBSCAN
        self.core_enhancement_features = [
            'is_dbscan_outlier',           # Binary risk flag
            'kmeans_segment_encoded',      # Segment value tier
            'segment_avg_session',         # Segment context
            'outlier_risk_interaction'     # Risk interaction score
        ]
    
    def load_core_data(self) -> pd.DataFrame:
        """Load only essential K-Means + DBSCAN data"""
        conn = psycopg2.connect(
            host="localhost",
            database="casino_research",
            user="researcher", 
            password="academic_password_2024"
        )
        
        # Optimized query - Only essential algorithms
        query = f"""
        SELECT 
            cf.customer_id,
            
            -- Base behavioral features
            cf.total_bet,
            cf.avg_bet, 
            cf.loss_rate,
            cf.total_sessions,
            cf.days_since_last_visit,
            cf.session_duration_volatility,
            cf.loss_chasing_score,
            cf.sessions_last_30d,
            cf.bet_trend_ratio,
            
            -- K-Means segment (primary business logic)
            ks.cluster_label as kmeans_segment,
            ks.cluster_id as kmeans_cluster_id,
            ks.avg_session_from_metadata as segment_avg_session,
            
            -- DBSCAN outlier detection (risk management)
            COALESCE(mas_dbscan.is_outlier, false) as is_dbscan_outlier,
            COALESCE(mas_dbscan.cluster_id, -1) as dbscan_cluster_id
            
        FROM casino_data.customer_features cf
        LEFT JOIN casino_data.kmeans_segments ks 
            ON cf.customer_id = ks.customer_id 
            AND ks.period_id = '{self.period_id}'
            AND ks.created_at >= '2025-07-21'
        LEFT JOIN casino_data.multi_algorithm_segments mas_dbscan
            ON cf.customer_id = mas_dbscan.customer_id
            AND mas_dbscan.period_id = '{self.period_id}'
            AND mas_dbscan.algorithm_name = 'dbscan'
        WHERE cf.analysis_period = '{self.period_id}'
        AND cf.total_bet > 0
        AND ks.cluster_label IS NOT NULL  -- Only customers with K-Means segments
        ORDER BY cf.customer_id
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(df)} customers with K-Means + DBSCAN data")
        return df
    
    def create_optimized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create focused, high-impact features"""
        df_optimized = df.copy()
        
        # === K-MEANS SEGMENT ENCODING ===
        # Convert segments to numerical hierarchy for RF
        segment_hierarchy = {
            'Casual_Player': 1,
            'Regular_Player': 2,
            'High_Value_Player': 3,
            'At_Risk_Player': 0  # Special treatment needed
        }
        df_optimized['kmeans_segment_encoded'] = df_optimized['kmeans_segment'].map(segment_hierarchy).fillna(0)
        
        # === DBSCAN RISK INTEGRATION ===
        # Simple but powerful binary flag
        df_optimized['is_dbscan_outlier'] = df_optimized['is_dbscan_outlier'].fillna(False).astype(int)
        
        # === RISK INTERACTION SCORE ===
        # Combines behavioral risk with outlier detection
        df_optimized['outlier_risk_interaction'] = (
            df_optimized['loss_chasing_score'] * 0.4 +
            df_optimized['is_dbscan_outlier'] * 30 +  # Heavy penalty for outliers
            (df_optimized['loss_rate'] > 25).astype(int) * 15 +
            (df_optimized['days_since_last_visit'] > 60).astype(int) * 10
        )
        
        # === SEGMENT CONTEXT FEATURES ===
        # How customer compares to their segment
        df_optimized['personal_vs_segment_ratio'] = (
            df_optimized['total_bet'] / df_optimized['segment_avg_session'].replace(0, 1)
        ).fillna(1.0)
        
        # === BUSINESS LOGIC FLAGS ===
        df_optimized['is_high_value'] = (df_optimized['kmeans_segment_encoded'] >= 2).astype(int)
        df_optimized['needs_intervention'] = (
            (df_optimized['is_dbscan_outlier'] == 1) | 
            (df_optimized['outlier_risk_interaction'] > 50)
        ).astype(int)
        
        logger.info("Created optimized K-Means + DBSCAN features")
        return df_optimized
    
    def create_business_promotion_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create business-focused promotion categories"""
        df_labeled = df.copy()
        
        promotion_labels = []
        
        for _, customer in df.iterrows():
            
            # INTERVENTION_NEEDED: Clear risk signals
            if customer['needs_intervention'] or customer['outlier_risk_interaction'] > 60:
                label = 'INTERVENTION_NEEDED'
            
            # VIP_TREATMENT: High value + stable behavior
            elif (customer['kmeans_segment_encoded'] == 3 and  # High_Value_Player
                  customer['is_dbscan_outlier'] == 0 and
                  customer['bet_trend_ratio'] > 0.8):
                label = 'VIP_TREATMENT'
            
            # GROWTH_TARGET: Regular customers with potential
            elif (customer['kmeans_segment_encoded'] == 2 and  # Regular_Player
                  customer['personal_vs_segment_ratio'] > 1.2 and
                  customer['is_dbscan_outlier'] == 0):
                label = 'GROWTH_TARGET'
            
            # STANDARD_PROMO: Safe casual players
            elif (customer['kmeans_segment_encoded'] == 1 and  # Casual_Player
                  customer['outlier_risk_interaction'] < 20 and
                  customer['total_sessions'] >= 3):
                label = 'STANDARD_PROMO'
            
            # NO_PROMOTION: Everyone else
            else:
                label = 'NO_PROMOTION'
                
            promotion_labels.append(label)
        
        df_labeled['promotion_label'] = promotion_labels
        
        # Log distribution for business validation
        label_dist = df_labeled['promotion_label'].value_counts()
        logger.info(f"Business promotion distribution: {label_dist.to_dict()}")
        
        return df_labeled
    
    def train_optimized_model(self, df: pd.DataFrame):
        """Train RF with focused feature set"""
        
        # Define feature set
        all_features = (
            self.base_behavioral_features + 
            self.core_enhancement_features + 
            ['personal_vs_segment_ratio', 'is_high_value', 'needs_intervention']
        )
        
        # Filter available features
        available_features = [col for col in all_features if col in df.columns]
        self.feature_names = available_features
        
        logger.info(f"Training with {len(available_features)} optimized features")
        
        # Prepare data
        X = df[available_features].fillna(0)
        y = df['promotion_label']
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train optimized Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=150,  # Slightly reduced for efficiency
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        
        self.rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        accuracy = self.rf_model.score(X_test_scaled, y_test)
        y_pred = self.rf_model.predict(X_test_scaled)
        
        logger.info(f"Optimized model accuracy: {accuracy:.4f}")
        
        # Feature importance analysis
        self.feature_importance_df = pd.DataFrame({
            'feature': available_features,
            'importance': self.rf_model.feature_importances_,
            'category': ['behavioral' if f in self.base_behavioral_features else 'enhancement' 
                        for f in available_features]
        }).sort_values('importance', ascending=False)
        
        # Business insights
        print("\n🎯 TOP 10 MOST IMPORTANT FEATURES:")
        print(self.feature_importance_df.head(10))
        
        # Category analysis
        behavioral_importance = self.feature_importance_df[
            self.feature_importance_df['category'] == 'behavioral'
        ]['importance'].sum()
        
        enhancement_importance = self.feature_importance_df[
            self.feature_importance_df['category'] == 'enhancement'
        ]['importance'].sum()
        
        print(f"\n  FEATURE CATEGORY ANALYSIS:")
        print(f"Behavioral Features: {behavioral_importance:.3f} ({behavioral_importance/(behavioral_importance+enhancement_importance)*100:.1f}%)")
        print(f"K-Means+DBSCAN Enhancement: {enhancement_importance:.3f} ({enhancement_importance/(behavioral_importance+enhancement_importance)*100:.1f}%)")
        
        # Classification report
        y_test_labels = self.label_encoder.inverse_transform(y_test)
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        
        print(f"\n BUSINESS CLASSIFICATION REPORT:")
        print(classification_report(y_test_labels, y_pred_labels))
        
        return {
            'accuracy': accuracy,
            'behavioral_importance': behavioral_importance,
            'enhancement_importance': enhancement_importance,
            'enhancement_contribution': enhancement_importance / (behavioral_importance + enhancement_importance),
            'feature_importance': self.feature_importance_df
        }
    
    def save_optimized_model(self, output_dir: str = 'models/optimized_rf'):
        """Save the streamlined model"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        model_path = f"{output_dir}/optimized_kmeans_dbscan_rf_{self.period_id}.pkl"
        
        model_package = {
            'rf_model': self.rf_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance_df,
            'period_id': self.period_id,
            'model_type': 'K-Means + DBSCAN Optimized',
            'training_date': datetime.now().isoformat()
        }
        
        joblib.dump(model_package, model_path)
        logger.info(f"Optimized model saved to {model_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', required=True, help='Period ID (e.g., 2022-H1)')
    parser.add_argument('--output_dir', default='models/optimized_rf', help='Output directory')
    args = parser.parse_args()
    
    logger.info(f"Training Optimized K-Means + DBSCAN RF model for {args.period}")
    
    # Initialize optimized model
    model = OptimizedRFPromotionModel(args.period)
    
    # Load core data
    df = model.load_core_data()
    
    if df.empty:
        logger.error(f"No data found for period {args.period}")
        return
    
    # Create optimized features
    df_optimized = model.create_optimized_features(df)
    
    # Create business promotion labels
    df_labeled = model.create_business_promotion_labels(df_optimized)
    
    # Train optimized model
    results = model.train_optimized_model(df_labeled)
    
    # Save model
    model.save_optimized_model(args.output_dir)
    
    print(f"\n{'='*70}")
    print(f" OPTIMIZED K-MEANS + DBSCAN RF TRAINING COMPLETED - {args.period}")
    print(f"{'='*70}")
    print(f" Accuracy: {results['accuracy']:.4f}")
    print(f" Enhancement Contribution: {results['enhancement_contribution']:.1%}")
    print(f" Model saved to: {args.output_dir}")
    print(f" Strategy: K-Means (stable segments) + DBSCAN (risk detection)")
    
    logger.info(" Optimized RF model training completed successfully!")

if __name__ == "__main__":
    main()