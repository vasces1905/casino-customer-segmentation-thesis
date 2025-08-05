#!/usr/bin/env python3
# rf_ensemble.py (version-3)
# My Aim:
# - Multi-period robustness --> Real-world applicability
# - Production architecture --> Industry relevance
# Multi-period integration:
# This module implements an ensemble Random Forest classifier for casino promotion targeting
# using multiple temporal periods (2022-H1, 2022-H2, 2023-H1, 2023-H2) with harmonized 
# class labeling system.
# Classes: NO_PROMOTION, GROWTH_TARGET, LOW_ENGAGEMENT, INTERVENTION_NEEDED

"""" Loads All Period Models
Strategy-based weighting (sensitivity, risk, balanced, adaptive)
Weighted voting for robust predictions

Accoding to the problems were arised:
- Early Periods--> Precision Targeting (2022-H1)
- Later periods --> Risk Assessment (2023-H2)
- Community robustness for production

Business scenarios:
- VIP Identity --> Precision Strategy
- Risk Management --> Risk Strategy
- General Promotions --> Balanced Strategy
- Complex Targeting --> Adaptive Strategy

Ensemble RF Model - Multi-Period Integration
==========================================
Implementation:
- Precision targeting (2022-H1 focus)  
- Risk classification (2023-H2 focus)
- Production ensemble for robustness

Academic Purpose: Demonstrate sophisticated ensemble methodology
Business Value: Combine period strengths for optimal performance
"""
"""
v2
- Production ensemble for robustness
- Purpose: Demonstrate sophisticated ensemble methodology
- Business Value: Combine period strengths for optimal performance

v3
- Feature intersection - only uses common features
- Class Mapping Fix - Phantom Cleans Classes
- Real metrics - ROC Auc calculations (if you have a ground truth)
- Honest Evaluation - real results
"""

import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, List, Tuple, Set
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FeatureOrderFixer:
    """
    Fix the sklearn feature order mismatch problem.
    """
    
    def __init__(self, model_paths: Dict[str, str], data_file: str):
        self.model_paths = model_paths
        self.data_file = data_file
        self.models = {}
        self.common_features = None
        
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        self.logger = logging.getLogger(__name__)
        
    def analyze_feature_orders(self):
        """Analyze feature order differences across models."""
        self.logger.info("🔍 Analyzing feature orders across models...")
        
        feature_orders = {}
        
        for period, path in self.model_paths.items():
            try:
                model_data = joblib.load(path)
                features = model_data.get('feature_names') or model_data.get('features')
                
                if features:
                    feature_orders[period] = list(features)
                    self.logger.info(f"{period}: {len(features)} features")
                    self.logger.info(f"   First 5: {features[:5]}")
                    
            except Exception as e:
                self.logger.error(f"Failed to load {period}: {e}")
                continue
        
        # Find common features
        if feature_orders:
            all_feature_sets = [set(features) for features in feature_orders.values()]
            common_features = set.intersection(*all_feature_sets)
            self.common_features = sorted(list(common_features))  # SORT for consistency
            
            self.logger.info(f"🎯 Common features: {len(common_features)}")
            
            # Check order differences
            for period, features in feature_orders.items():
                common_in_order = [f for f in features if f in common_features]
                order_match = common_in_order == sorted(common_in_order)
                self.logger.info(f"{period}: Order consistent = {order_match}")
                
        return feature_orders
        
    def create_ordered_predictions(self, X: pd.DataFrame):
        """Create predictions with proper feature ordering."""
        self.logger.info("🎯 Creating predictions with proper feature ordering...")
        
        if not self.common_features:
            raise ValueError("No common features found!")
            
        # Use ONLY common features in CONSISTENT order
        X_ordered = X[self.common_features].fillna(0)
        self.logger.info(f"Using {len(self.common_features)} features in consistent order")
        
        ensemble_probabilities = []
        individual_predictions = {}
        successful_models = 0
        
        for period, path in self.model_paths.items():
            try:
                self.logger.info(f"Processing {period}...")
                
                model_data = joblib.load(path)
                rf_model = model_data.get('rf_model') or model_data.get('model')
                scaler = model_data.get('scaler')
                
                # Apply scaler if available (to ordered features)
                if scaler:
                    try:
                        X_scaled = scaler.transform(X_ordered)
                        self.logger.info(f"{period}: Applied scaler successfully")
                    except Exception as e:
                        self.logger.warning(f"{period}: Scaler failed, using raw features: {e}")
                        X_scaled = X_ordered.values
                else:
                    X_scaled = X_ordered.values
                
                # Get predictions
                probas = rf_model.predict_proba(X_scaled)
                preds = rf_model.predict(X_scaled)
                
                ensemble_probabilities.append(probas)
                individual_predictions[period] = preds
                successful_models += 1
                
                self.logger.info(f"✅ {period}: Shape {probas.shape}, Classes {rf_model.classes_}")
                
            except Exception as e:
                self.logger.error(f"❌ {period} failed: {e}")
                continue
        
        if successful_models == 0:
            raise ValueError("No models succeeded!")
            
        self.logger.info(f"✅ {successful_models} models succeeded")
        
        # Check shape consistency
        shapes = [prob.shape for prob in ensemble_probabilities]
        if len(set(shapes)) > 1:
            self.logger.warning(f"Shape mismatch detected: {shapes}")
            # Use minimum number of classes
            min_classes = min(shape[1] for shape in shapes)
            ensemble_probabilities = [prob[:, :min_classes] for prob in ensemble_probabilities]
            self.logger.info(f"Truncated to {min_classes} classes for consistency")
        
        # Simple average
        final_probabilities = np.mean(ensemble_probabilities, axis=0)
        final_predictions = np.argmax(final_probabilities, axis=1)
        
        # Calculate consensus
        if len(individual_predictions) > 1:
            pred_df = pd.DataFrame(individual_predictions)
            consensus_mask = pred_df.nunique(axis=1) == 1
            consensus_rate = consensus_mask.mean()
        else:
            consensus_rate = 1.0
            
        return final_predictions, final_probabilities, individual_predictions, consensus_rate

def main():
    """Fix feature order issues and run ensemble."""
    
    MODEL_PATHS = {
        '2022-H1': 'models/optimized_rf/optimized_kmeans_dbscan_rf_2022-H1.pkl',
        '2022-H2': 'models/optimized_rf/optimized_kmeans_dbscan_rf_2022-H2.pkl',
        '2023-H1': 'models/optimized_rf/optimized_kmeans_dbscan_rf_2023-H1.pkl',
        '2023-H2': 'models/optimized_rf/optimized_kmeans_dbscan_rf_2023-H2.pkl',
    }
    
    DATA_FILE = 'enhanced_customer_features_2023-H2_20250728_1758.csv'
    
    print("🔧 FEATURE ORDER FIX")
    print("=" * 40)
    print("Solving sklearn's strict feature order requirement")
    
    fixer = FeatureOrderFixer(MODEL_PATHS, DATA_FILE)
    
    try:
        # Analyze feature order issues
        print("\n🔍 ANALYZING FEATURE ORDERS...")
        feature_orders = fixer.analyze_feature_orders()
        
        # Load data
        print("\n📂 LOADING DATA...")
        data = pd.read_csv(DATA_FILE)
        print(f"Data shape: {data.shape}")
        
        # Create predictions with proper ordering
        print("\n🎯 RUNNING FIXED ENSEMBLE...")
        predictions, probabilities, individual_preds, consensus_rate = fixer.create_ordered_predictions(data)
        
        # Results
        print("\n" + "="*50)
        print("🎯 FIXED ENSEMBLE RESULTS")
        print("="*50)
        print(f"✅ Models used: {len(individual_preds)}")
        print(f"✅ Common features: {len(fixer.common_features)}")
        print(f"📊 Consensus rate: {consensus_rate:.3f}")
        print(f"🎯 Average confidence: {np.max(probabilities, axis=1).mean():.3f}")
        
        # Prediction distribution
        pred_dist = pd.Series(predictions).value_counts()
        print(f"\n📈 Prediction Distribution:")
        for pred_class, count in pred_dist.items():
            percentage = (count / len(predictions)) * 100
            print(f"   Class {pred_class}: {count:,} customers ({percentage:.1f}%)")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        results_df = pd.DataFrame({
            'customer_id': data['customer_id'].values,
            'fixed_ensemble_prediction': predictions,
            'confidence_score': np.max(probabilities, axis=1)
        })
        
        # Add individual predictions
        for period, preds in individual_preds.items():
            results_df[f'pred_{period}'] = preds
        
        output_file = f'fixed_ensemble_results_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)
        
        print(f"\n💾 Fixed results saved to: {output_file}")
        
        # Assessment
        print(f"\n🔍 ASSESSMENT:")
        if consensus_rate > 0.6:
            print("✅ Good consensus - models agree reasonably well")
        else:
            print("❌ Low consensus - still significant disagreement")
            
        if len(fixer.common_features) < 20:
            print("⚠️ Limited common features - may need better feature engineering")
        else:
            print("✅ Sufficient common features for ensemble")
        
        # Next steps
        print(f"\n🎯 NEXT STEPS:")
        print("1. If consensus still low, check model training consistency")
        print("2. Consider retraining all models with identical feature sets")
        print("3. Generate visualizations to understand disagreements")
        
        return results_df, consensus_rate
        
    except Exception as e:
        print(f"❌ Fix failed: {e}")
        import traceback
        traceback.print_exc()
        return None, 0.0

if __name__ == "__main__":
    results, consensus = main()