#!/usr/bin/env python3
"""
Random Forest Class Imbalance Fix
Problem: 27.1:1 class imbalance causing single class prediction
Solution: Aggressive SMOTE + Balanced RF parameters
"""

import pandas as pd
import numpy as np
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import joblib
import os

def load_balanced_data(period='2022-H1'):
    """
    Load customer features + promo labels with proper joins
    """
    print(f"Loading data for {period}")
    
    conn = psycopg2.connect(
        host="localhost",
        database="casino_research",
        user="researcher",
        password="academic_password_2024"
    )
    
    # Use customer_features_robust (35 columns) for best feature set
    query = f"""
    SELECT cf.*, pl.promo_label
    FROM casino_data.customer_features_robust cf
    JOIN casino_data.promo_label pl ON cf.customer_id = pl.customer_id 
    WHERE pl.period = '{period}';
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Loaded {len(df)} records with {df.shape[1]} columns")
    
    # Prepare features
    feature_cols = [col for col in df.columns 
                   if col not in ['customer_id', 'promo_label', 'analysis_period']]
    
    X = df[feature_cols].select_dtypes(include=[np.number])  # Numeric only
    y = df['promo_label']
    
    print(f"Features: {X.shape[1]} numeric columns")
    print(f"Original class distribution: {Counter(y)}")
    
    return X, y, df

def apply_aggressive_balancing(X, y):
    """
    Apply aggressive class balancing for 27:1 imbalance
    """
    print("\nApplying aggressive class balancing...")
    
    # Strategy 1: SMOTE to oversample minorities
    smote = SMOTE(
        sampling_strategy='auto',  # Balance all minorities
        k_neighbors=min(5, min(Counter(y).values()) - 1),  # Adaptive k_neighbors
        random_state=42
    )
    
    # Strategy 2: RandomUnderSampler to reduce majority
    under_sampler = RandomUnderSampler(
        sampling_strategy='auto',  # Balance after SMOTE
        random_state=42
    )
    
    # Combined pipeline
    balancing_pipeline = ImbPipeline([
        ('over', smote),
        ('under', under_sampler)
    ])
    
    X_balanced, y_balanced = balancing_pipeline.fit_resample(X, y)
    
    print(f"Balanced class distribution: {Counter(y_balanced)}")
    
    # Calculate final imbalance ratio
    counts = list(Counter(y_balanced).values())
    new_ratio = max(counts) / min(counts)
    print(f"New imbalance ratio: {new_ratio:.1f}:1")
    
    return X_balanced, y_balanced, balancing_pipeline

def create_balanced_rf_model(y_train):
    """
    Create Random Forest optimized for balanced multi-class prediction
    """
    # Compute class weights
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"Class weights: {class_weight_dict}")
    
    # Optimized RF for balanced prediction
    rf = RandomForestClassifier(
        n_estimators=500,                    # More trees for stability
        max_depth=None,                      # No depth limit
        min_samples_split=2,                 # Allow fine splits
        min_samples_leaf=1,                  # Single sample leaves OK
        max_features='sqrt',                 # Feature randomness
        class_weight=class_weight_dict,      # CRITICAL: Balanced weights
        bootstrap=True,
        oob_score=True,                      # Out-of-bag validation
        random_state=42,
        n_jobs=-1                            # Use all cores
    )
    
    return rf

def train_and_evaluate_model(period='2022-H1'):
    """
    Complete training pipeline with imbalance fix
    """
    print(f"\nTRAINING BALANCED RF MODEL - {period}")
    print("=" * 60)
    
    # Load data
    X, y, df_full = load_balanced_data(period)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply aggressive balancing
    X_balanced, y_balanced, balancing_pipeline = apply_aggressive_balancing(X_scaled, y_encoded)
    
    # Train-test split with balanced data
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, 
        test_size=0.2, 
        stratify=y_balanced,  # Maintain balance in split
        random_state=42
    )
    
    # Create and train balanced RF
    rf = create_balanced_rf_model(y_train)
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)
    
    # CRITICAL: Check prediction diversity
    pred_distribution = Counter(y_pred)
    total_classes = len(le.classes_)
    predicted_classes = len(pred_distribution)
    
    print(f"\nMODEL PERFORMANCE:")
    print(f"   Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"   OOB Score: {rf.oob_score_:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf, X_balanced, y_balanced, cv=5, scoring='accuracy')
    print(f"   CV Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    print(f"\nPREDICTION DIVERSITY CHECK:")
    print(f"   Expected Classes: {total_classes}")
    print(f"   Predicted Classes: {predicted_classes}")
    
    # Show prediction distribution
    print(f"\nPREDICTION DISTRIBUTION:")
    for class_idx, count in pred_distribution.items():
        class_name = le.inverse_transform([class_idx])[0]
        percentage = (count / len(y_pred)) * 100
        print(f"   {class_name}: {count} ({percentage:.1f}%)")
    
    # Success check
    if predicted_classes == total_classes:
        print(f"\n✅ SUCCESS: All {total_classes} classes predicted!")
        print("✅ Multi-class prediction problem SOLVED!")
    else:
        missing_classes = total_classes - predicted_classes
        print(f"\n⚠️  PARTIAL SUCCESS: {missing_classes} classes still missing")
        print("   Try more aggressive balancing parameters")
    
    # Detailed classification report
    print(f"\nCLASSIFICATION REPORT:")
    class_names = le.classes_
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTOP 10 MOST IMPORTANT FEATURES:")
    for _, row in feature_importance.head(10).iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    model_data = {
        'model': rf,
        'scaler': scaler,
        'label_encoder': le,
        'balancing_pipeline': balancing_pipeline,
        'feature_names': list(X.columns),
        'class_names': list(class_names),
        'period': period,
        'accuracy': accuracy_score(y_test, y_pred),
        'cv_scores': cv_scores,
        'all_classes_predicted': (predicted_classes == total_classes),
        'prediction_distribution': dict(pred_distribution),
        'original_imbalance_ratio': 27.1,  # From diagnosis
        'balanced_imbalance_ratio': max(Counter(y_balanced).values()) / min(Counter(y_balanced).values())
    }
    
    # Create models directory if needed
    os.makedirs('models/balanced/', exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M')
    filename = f"models/balanced/balanced_rf_{period}_{timestamp}.pkl"
    joblib.dump(model_data, filename)
    
    print(f"\n💾 Model saved: {filename}")
    
    return model_data

def test_all_periods():
    """
    Test balanced RF on all periods
    """
    print("TESTING BALANCED RF ON ALL PERIODS")
    print("=" * 50)
    
    periods = ['2022-H1', '2022-H2', '2023-H1', '2023-H2']
    results = {}
    
    for period in periods:
        try:
            print(f"\n{'='*20} {period} {'='*20}")
            model_data = train_and_evaluate_model(period)
            results[period] = model_data['all_classes_predicted']
            
        except Exception as e:
            print(f"❌ Error in {period}: {e}")
            results[period] = False
    
    print(f"\n📊 FINAL RESULTS SUMMARY:")
    print("=" * 40)
    for period, success in results.items():
        status = "✅ SUCCESS" if success else "⚠️  PARTIAL"
        print(f"   {period}: {status}")
    
    successful_periods = sum(results.values())
    print(f"\n🎯 University of Bath Standard:")
    print(f"   {successful_periods}/{len(periods)} periods achieving multi-class prediction")
    
    if successful_periods >= 3:
        print("🎓 THESIS STANDARD: ACHIEVED!")
    else:
        print("📈 PROGRESS: Need fine-tuning for remaining periods")

if __name__ == "__main__":
    # Test single period first
    print("UNIVERSITY OF BATH - RF CLASS IMBALANCE FIX")
    print("=" * 60)
    
    # Start with most problematic period (2022-H1 with 27:1 imbalance)
    model_data = train_and_evaluate_model('2022-H1')
    
    if model_data['all_classes_predicted']:
        print("\n🚀 Single period SUCCESS - Testing all periods...")
        test_all_periods()
    else:
        print("\n🔧 Need parameter tuning - Check balancing strategy")