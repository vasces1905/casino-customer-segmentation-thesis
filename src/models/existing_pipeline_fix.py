
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
    
    print(f"\nPREDICTION CHECK:")
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
