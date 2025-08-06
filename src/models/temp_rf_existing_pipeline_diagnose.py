#!/usr/bin/env python3
"""
Existing Pipeline Diagnosis for Multi-Class Prediction Issues
Problem: Random Forest models achieving high accuracy but predicting only 1-2 classes
Solution: Diagnose current pipeline and provide targeted fixes
"""

import pandas as pd
import numpy as np
import psycopg2
import joblib
from collections import Counter
import os

def diagnose_existing_models():
    """
    Analyze existing PKL model files
    """
    print("EXISTING MODEL DIAGNOSIS")
    print("=" * 50)
    
    models_dir = "models/active/"
    
    if not os.path.exists(models_dir):
        print(f"Models directory not found: {models_dir}")
        return
    
    pkl_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    print(f"Found {len(pkl_files)} PKL files:")
    for pkl in pkl_files:
        print(f"   - {pkl}")
    
    for pkl_file in pkl_files:
        try:
            print(f"\nANALYZING: {pkl_file}")
            print("-" * 40)
            
            model_path = os.path.join(models_dir, pkl_file)
            model_data = joblib.load(model_path)
            
            print("MODEL CONTENTS:")
            for key in model_data.keys():
                print(f"   - {key}: {type(model_data[key])}")
            
            if 'model' in model_data:
                rf_model = model_data['model']
                print(f"   - RF Trees: {rf_model.n_estimators}")
                print(f"   - RF Classes: {rf_model.classes_}")
                print(f"   - RF Features: {rf_model.n_features_in_}")
                
                if hasattr(rf_model, 'class_weight'):
                    print(f"   - Class Weight: {rf_model.class_weight}")
                else:
                    print("   No class_weight found")
            
            if 'label_encoder' in model_data:
                le = model_data['label_encoder']
                print(f"   - Classes: {le.classes_}")
            
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")

def diagnose_prediction_issue(period='2022-H1'):
    """
    Diagnose prediction distribution issues
    """
    print(f"\nPREDICTION DIAGNOSIS - {period}")
    print("=" * 50)
    
    conn = psycopg2.connect(
        host="localhost",
        database="casino_research",
        user="researcher",
        password="academic_password_2024"
    )
    
    query = f"""
    SELECT cf.*, pl.promo_label
    FROM customer_features cf
    JOIN promo_label pl ON cf.customer_id = pl.customer_id 
    WHERE pl.period = '{period}'
    LIMIT 1000;
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    label_dist = Counter(df['promo_label'])
    print(f"ACTUAL LABEL DISTRIBUTION:")
    total = len(df)
    for label, count in label_dist.items():
        pct = (count/total)*100
        print(f"   {label}: {count} ({pct:.1f}%)")
    
    model_file = f"models/active/harmonized_rf_{period}_latest.pkl"
    
    if os.path.exists(model_file):
        try:
            model_data = joblib.load(model_file)
            
            feature_cols = [col for col in df.columns 
                           if col not in ['customer_id', 'promo_label', 'feature_created_at', 'analysis_period']]
            X = df[feature_cols]
            
            if 'scaler' in model_data:
                X_scaled = model_data['scaler'].transform(X)
            else:
                X_scaled = X.values
            
            rf_model = model_data['model']
            predictions = rf_model.predict(X_scaled)
            
            pred_dist = Counter(predictions)
            print(f"\nMODEL PREDICTIONS:")
            for pred, count in pred_dist.items():
                pct = (count/len(predictions))*100
                
                if 'label_encoder' in model_data:
                    class_name = model_data['label_encoder'].inverse_transform([pred])[0]
                    print(f"   {class_name}: {count} ({pct:.1f}%)")
                else:
                    print(f"   Class {pred}: {count} ({pct:.1f}%)")
            
            unique_predictions = len(pred_dist)
            total_classes = len(model_data.get('label_encoder', {}).classes_ if 'label_encoder' in model_data else rf_model.classes_)
            
            print(f"\nPROBLEM ANALYSIS:")
            print(f"   Expected Classes: {total_classes}")
            print(f"   Predicted Classes: {unique_predictions}")
            
            if unique_predictions < total_classes:
                print(f"   PROBLEM DETECTED: Model only predicts {unique_predictions}/{total_classes} classes")
                
                max_count = max(label_dist.values())
                min_count = min(label_dist.values())
                imbalance_ratio = max_count / min_count
                print(f"   Data Imbalance Ratio: {imbalance_ratio:.1f}:1")
                
                if imbalance_ratio > 10:
                    print("   HIGH IMBALANCE - Causes single class prediction")
            
        except Exception as e:
            print(f"Error testing model: {e}")
    else:
        print(f"Model file not found: {model_file}")

def generate_existing_pipeline_fix():
    """
    Generate fix code for existing rf_clean_harmonized_training.py
    """
    print(f"\nEXISTING PIPELINE FIX")
    print("=" * 50)
    
    fix_code = '''
# PATCH FOR rf_clean_harmonized_training.py
# Add these modifications to your existing training script

# 1. IMPORT ADDITIONS (add to top of file)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.utils.class_weight import compute_class_weight

# 2. CLASS BALANCING FUNCTION (add after imports)
def apply_class_balancing(X, y, strategy='aggressive'):
    """
    Apply class balancing to fix single-class prediction
    """
    print(f"Original distribution: {Counter(y)}")
    
    if strategy == 'aggressive':
        over_sampler = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
        under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
        
        pipeline = ImbPipeline([
            ('over', over_sampler),
            ('under', under_sampler)
        ])
        
        X_balanced, y_balanced = pipeline.fit_resample(X, y)
        
    elif strategy == 'smote_only':
        smote = SMOTE(sampling_strategy='auto', k_neighbors=3, random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
    else:
        X_balanced, y_balanced = X, y
    
    print(f"Balanced distribution: {Counter(y_balanced)}")
    return X_balanced, y_balanced

# 3. RF PARAMETERS FIX (replace existing RF initialization)
def create_balanced_rf(y_train):
    """
    Create RF with proper class balancing parameters
    """
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight=class_weight_dict,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    
    return rf

# 4. MODIFY YOUR TRAINING SECTION
def train_balanced_model(period):
    """
    Modified training function with class balancing
    """
    # Load your data (existing code)
    # ... your existing data loading code ...
    
    # Apply preprocessing (existing code)
    # ... your existing preprocessing ...
    
    # NEW: Apply class balancing
    X_balanced, y_balanced = apply_class_balancing(X_scaled, y_encoded, strategy='aggressive')
    
    # Train-test split with balanced data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, 
        test_size=0.2, 
        stratify=y_balanced,
        random_state=42
    )
    
    # Create balanced RF
    rf = create_balanced_rf(y_train)
    
    # Train model
    rf.fit(X_train, y_train)
    
    # Test predictions
    y_pred = rf.predict(X_test)
    
    # Check prediction distribution
    pred_dist = Counter(y_pred)
    total_classes = len(np.unique(y_balanced))
    predicted_classes = len(pred_dist)
    
    print(f"\\nPREDICTION CHECK:")
    print(f"   Total Classes: {total_classes}")
    print(f"   Predicted Classes: {predicted_classes}")
    
    for class_val, count in pred_dist.items():
        class_name = label_encoder.inverse_transform([class_val])[0]
        pct = (count/len(y_pred))*100
        print(f"   {class_name}: {count} ({pct:.1f}%)")
    
    if predicted_classes < total_classes:
        print(f"   WARNING: Still missing {total_classes - predicted_classes} classes")
        print(f"   Try more aggressive balancing or different parameters")
    else:
        print(f"   SUCCESS: All {total_classes} classes predicted")
    
    # Save model (existing code with additions)
    model_data = {
        'model': rf,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'class_weights': dict(zip(np.unique(y_train), compute_class_weight('balanced', classes=np.unique(y_train), y=y_train))),
        'feature_names': feature_columns,
        'period': period,
        'balanced_distribution': Counter(y_balanced),
        'prediction_distribution': pred_dist,
        'all_classes_predicted': (predicted_classes == total_classes)
    }
    
    # Save with timestamp
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    filename = f"models/active/balanced_rf_{period}_{timestamp}.pkl"
    joblib.dump(model_data, filename)
    
    return model_data

# 5. USAGE - Modify your main section:
if __name__ == "__main__":
    period = args.period
    model_data = train_balanced_model(period)
    
    if model_data['all_classes_predicted']:
        print("University of Bath Standard: ACHIEVED")
    else:
        print("Need more balancing - try different parameters")
'''
    
    return fix_code

def main():
    """
    Main diagnosis and fix generation function
    """
    print("UNIVERSITY OF BATH - EXISTING PIPELINE DIAGNOSIS")
    print("=" * 60)
    
    diagnose_existing_models()
    
    periods = ['2022-H1', '2022-H2', '2023-H1', '2023-H2']
    for period in periods:
        try:
            diagnose_prediction_issue(period)
        except Exception as e:
            print(f"Error diagnosing {period}: {e}")
    
    fix_code = generate_existing_pipeline_fix()
    
    with open('existing_pipeline_fix.py', 'w') as f:
        f.write(fix_code)
    
    print(f"\nFIX CODE GENERATED:")
    print(f"   File: existing_pipeline_fix.py")
    print(f"   Apply patches to rf_clean_harmonized_training.py")
    print(f"   Keep your existing workflow")
    
    print(f"\nNEXT STEPS:")
    print(f"   1. Run this diagnosis first")
    print(f"   2. Apply patches to rf_clean_harmonized_training.py")
    print(f"   3. Re-train models with balancing")
    print(f"   4. Test with rf_baseline_model_comparison_allPeriods.py")
    print(f"   5. Verify 6-class predictions")

if __name__ == "__main__":
    main()