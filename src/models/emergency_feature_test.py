
# EMERGENCY FEATURE CREATION (for testing only)
# This creates basic features to test RF balancing

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def create_emergency_features(period='2022-H1'):
    """
    Create emergency features for RF testing
    """
    conn = psycopg2.connect(
        host="localhost",
        database="casino_research", 
        user="researcher",
        password="academic_password_2024"
    )
    
    # Get promo labels
    df = pd.read_sql(f"""
        SELECT customer_id, promo_label, risk_score, value_score, engagement_score
        FROM casino_data.promo_label 
        WHERE period = '{period}';
    """, conn)
    
    conn.close()
    
    # Create synthetic features based on existing scores
    np.random.seed(42)
    n_customers = len(df)
    
    # Generate realistic casino features
    features_df = pd.DataFrame({
        'customer_id': df['customer_id'],
        'total_bet': np.random.lognormal(8, 1.5, n_customers),
        'total_sessions': np.random.randint(1, 50, n_customers),
        'avg_bet': np.random.lognormal(4, 1, n_customers),
        'loss_rate': np.random.beta(2, 3, n_customers),
        'days_since_last_visit': np.random.randint(0, 90, n_customers),
        'game_diversity': np.random.randint(1, 10, n_customers),
        'weekend_preference': np.random.random(n_customers),
        'late_night_player': np.random.random(n_customers),
        'risk_score': df['risk_score'],
        'value_score': df['value_score'], 
        'engagement_score': df['engagement_score']
    })
    
    # Add promo_label
    features_df['promo_label'] = df['promo_label']
    
    return features_df

def test_rf_with_emergency_features(period='2022-H1'):
    """
    Test RF balancing with emergency features
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from imblearn.over_sampling import SMOTE
    from collections import Counter
    
    # Create features
    df = create_emergency_features(period)
    
    # Prepare data
    feature_cols = [col for col in df.columns if col not in ['customer_id', 'promo_label']]
    X = df[feature_cols]
    y = df['promo_label']
    
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print(f"Original distribution: {Counter(y)}")
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X, y_encoded)
    
    print(f"Balanced distribution: {Counter(y_balanced)}")
    
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
    )
    
    # Train RF with balancing
    rf = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    # Check predictions
    pred_dist = Counter(y_pred)
    total_classes = len(le.classes_)
    predicted_classes = len(pred_dist)
    
    print(f"\nPREDICTION RESULTS:")
    print(f"Total classes: {total_classes}")
    print(f"Predicted classes: {predicted_classes}")
    
    for class_idx, count in pred_dist.items():
        class_name = le.inverse_transform([class_idx])[0]
        pct = (count/len(y_pred))*100
        print(f"   {class_name}: {count} ({pct:.1f}%)")
    
    if predicted_classes == total_classes:
        print("\nSUCCESS: All classes predicted!")
        print("The balancing technique works - now apply to real features")
    else:
        print(f"\nSTILL MISSING: {total_classes - predicted_classes} classes")
        print("Need more aggressive balancing")

if __name__ == "__main__":
    test_rf_with_emergency_features('2022-H1')
