#!/usr/bin/env python3
"""
Model Inspector - PKL File Analysis
=================================
Inspect saved .pkl models and analyze performance
"""

import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime
import os
import glob

def inspect_model(model_path: str):
    """Inspect a saved .pkl model"""
    print(f"INSPECTING MODEL: {os.path.basename(model_path)}")
    print("="*70)
    
    try:
        # Load model package
        model_package = joblib.load(model_path)
        
        # === BASIC INFO ===
        print(f" BASIC INFORMATION:")
        print(f"   Period: {model_package.get('period_id', 'N/A')}")
        print(f"   Strategy: {model_package.get('labeling_strategy', 'N/A')}")
        print(f"   Version: {model_package.get('model_version', 'N/A')}")
        print(f"   Training Date: {model_package.get('training_date', 'N/A')}")
        print(f"   Features Count: {model_package.get('n_features', 'N/A')}")
        print(f"   Classes Count: {model_package.get('n_classes', 'N/A')}")
        
        # === PERFORMANCE METRICS ===
        performance = model_package.get('performance_metrics', {})
        print(f"\n PERFORMANCE METRICS:")
        for metric, value in performance.items():
            print(f"   {metric}: {value}")
        
        # === FEATURE IMPORTANCE ===
        feature_importance = model_package.get('feature_importance')
        if feature_importance is not None:
            print(f"\n🔝 TOP 10 MOST IMPORTANT FEATURES:")
            print(feature_importance.head(10).to_string(index=False))
        
        # === MODEL DETAILS ===
        rf_model = model_package.get('rf_model')
        if rf_model is not None:
            print(f"\n MODEL CONFIGURATION:")
            print(f"   Estimators: {rf_model.n_estimators}")
            print(f"   Max Depth: {rf_model.max_depth}")
            print(f"   Min Samples Split: {rf_model.min_samples_split}")
            print(f"   Min Samples Leaf: {rf_model.min_samples_leaf}")
            print(f"   Max Features: {rf_model.max_features}")
        
        # === LABEL ENCODER ===
        label_encoder = model_package.get('label_encoder')
        if label_encoder is not None and hasattr(label_encoder, 'classes_'):
            print(f"\n LABEL CLASSES:")
            for i, class_name in enumerate(label_encoder.classes_):
                print(f"   {i}: {class_name}")
        
        # === ACADEMIC METADATA ===
        academic_metadata = model_package.get('academic_metadata', {})
        if academic_metadata:
            print(f"\n🎓 ACADEMIC METADATA:")
            for key, value in academic_metadata.items():
                print(f"   {key}: {value}")
        
        return model_package
        
    except Exception as e:
        print(f"Error inspecting model: {e}")
        return None

def compare_models(model_dir: str = "models/generic_rf"):
    """Compare all models in directory"""
    print(f"\n MODEL COMPARISON")
    print("="*70)
    
    # Find all model files
    model_files = glob.glob(f"{model_dir}/**/generic_rf_*.pkl", recursive=True)
    
    if not model_files:
        print(f"Caution! No models found in {model_dir}")
        return
    
    comparison_data = []
    
    for model_file in model_files:
        try:
            model_package = joblib.load(model_file)
            performance = model_package.get('performance_metrics', {})
            
            comparison_data.append({
                'File': os.path.basename(model_file),
                'Period': model_package.get('period_id', 'N/A'),
                'Strategy': model_package.get('labeling_strategy', 'N/A'),
                'Version': model_package.get('model_version', 'N/A'),
                'Accuracy': performance.get('accuracy', 'N/A'),
                'ROC_AUC': performance.get('roc_auc', performance.get('roc_auc_macro', 'N/A')),
                'CV_Mean': performance.get('cv_score_mean', 'N/A'),
                'Training_Date': model_package.get('training_date', 'N/A')[:10] if model_package.get('training_date') else 'N/A'
            })
            
        except Exception as e:
            print(f"Warning! Could not load {model_file}: {e}")
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        print(df_comparison.to_string(index=False))
        
        # Find best model
        if len(df_comparison) > 1:
            try:
                # Convert accuracy to numeric for comparison
                df_comparison['Accuracy_Numeric'] = pd.to_numeric(df_comparison['Accuracy'], errors='coerce')
                best_model = df_comparison.loc[df_comparison['Accuracy_Numeric'].idxmax()]
                print(f"\n Good! BEST MODEL (by accuracy):")
                print(f"   File: {best_model['File']}")
                print(f"   Accuracy: {best_model['Accuracy']}")
                print(f"   Strategy: {best_model['Strategy']}")
            except:
                pass

def validate_model_prediction(model_path: str, sample_data: dict = None):
    """Validate model can make predictions"""
    print(f"\n MODEL PREDICTION TEST")
    print("="*50)
    
    try:
        model_package = joblib.load(model_path)
        rf_model = model_package['rf_model']
        scaler = model_package['scaler']
        label_encoder = model_package['label_encoder']
        feature_names = model_package['feature_names']
        
        # Create sample data if not provided
        if sample_data is None:
            print("Creating sample customer data...")
            sample_data = {
                'total_bet': 1500.0,
                'avg_bet': 25.0,
                'loss_rate': 15.5,
                'total_sessions': 8,
                'days_since_last_visit': 5,
                'session_duration_volatility': 0.3,
                'loss_chasing_score': 12.0,
                'sessions_last_30d': 6,
                'bet_trend_ratio': 1.1,
                'kmeans_cluster_id': 2,
                'kmeans_segment_encoded': 2,
                'segment_avg_session': 187.5,
                'silhouette_score_customer': 0.45,
                'personal_vs_segment_ratio': 1.2,
                'risk_score': 18.5,
                'value_tier': 1,  # Encoded
                'engagement_level': 2,  # Encoded
                'is_high_value': 1,
                'needs_attention': 0,
                'segment_outperformer': 1
            }
        
        # Create feature vector
        feature_vector = []
        for feature_name in feature_names:
            feature_vector.append(sample_data.get(feature_name, 0))
        
        # Scale and predict
        X_sample = np.array([feature_vector])
        X_sample_scaled = scaler.transform(X_sample)
        
        prediction = rf_model.predict(X_sample_scaled)[0]
        prediction_proba = rf_model.predict_proba(X_sample_scaled)[0]
        
        # Decode prediction
        predicted_label = label_encoder.inverse_transform([prediction])[0]
        
        print(f"Done! MODEL PREDICTION SUCCESSFUL:")
        print(f"   Predicted Label: {predicted_label}")
        print(f"   Prediction Confidence:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"      {class_name}: {prediction_proba[i]:.3f}")
        
        return True
        
    except Exception as e:
        print(f"Fail! Prediction test failed: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Model Inspector')
    parser.add_argument('--model', help='Path to specific model file')
    parser.add_argument('--dir', default='models/generic_rf', help='Models directory')
    parser.add_argument('--compare', action='store_true', help='Compare all models')
    parser.add_argument('--test', action='store_true', help='Test model prediction')
    args = parser.parse_args()
    
    if args.model:
        # Inspect specific model
        model_package = inspect_model(args.model)
        
        if args.test and model_package:
            validate_model_prediction(args.model)
    
    elif args.compare:
        # Compare all models
        compare_models(args.dir)
    
    else:
        # Find and inspect latest models
        model_files = glob.glob(f"{args.dir}/**/generic_rf_*.pkl", recursive=True)
        
        if not model_files:
            print(f"Error! No models found in {args.dir}")
            return
        
        # Sort by modification time (newest first)
        model_files.sort(key=os.path.getmtime, reverse=True)
        
        print(f"FOUND {len(model_files)} MODELS")
        print(f"Inspecting latest model: {os.path.basename(model_files[0])}")
        
        model_package = inspect_model(model_files[0])
        
        if args.test and model_package:
            validate_model_prediction(model_files[0])
        
        if len(model_files) > 1:
            print(f"\n Use --compare to see all models comparison")
            print(f"Use --model PATH to inspect specific model")

if __name__ == "__main__":
    main()