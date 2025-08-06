#!/usr/bin/env python3
"""
Find Missing Customer Features Tables
The customer_features table is missing - let's find where the feature data is
"""

import pandas as pd
import psycopg2
import os

def find_all_tables():
    """
    Get complete list of all tables in the database
    """
    print("COMPLETE DATABASE TABLE LIST")
    print("=" * 40)
    
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="casino_research",
            user="researcher",
            password="academic_password_2024"
        )
        
        cursor = conn.cursor()
        
        # Get ALL tables from casino_data schema
        cursor.execute("""
            SELECT table_name, 
                   (SELECT COUNT(*) FROM information_schema.columns 
                    WHERE table_name = t.table_name AND table_schema = 'casino_data') as column_count
            FROM information_schema.tables t
            WHERE table_schema = 'casino_data' 
            ORDER BY table_name;
        """)
        
        tables = cursor.fetchall()
        
        print(f"FOUND {len(tables)} TABLES:")
        for table_name, col_count in tables:
            print(f"   - {table_name} ({col_count} columns)")
            
            # Check if this might be a features table
            if 'customer' in table_name.lower() or 'feature' in table_name.lower():
                print(f"     *** POTENTIAL FEATURES TABLE ***")
                
                # Get sample data
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM casino_data.{table_name};")
                    row_count = cursor.fetchone()[0]
                    print(f"     Records: {row_count}")
                    
                    # Show columns
                    cursor.execute(f"""
                        SELECT column_name, data_type 
                        FROM information_schema.columns 
                        WHERE table_name = '{table_name}' AND table_schema = 'casino_data'
                        ORDER BY ordinal_position
                        LIMIT 10;
                    """)
                    columns = cursor.fetchall()
                    print(f"     Sample columns:")
                    for col_name, col_type in columns:
                        print(f"       - {col_name}: {col_type}")
                        
                except Exception as e:
                    print(f"     Error checking {table_name}: {e}")
        
        cursor.close()
        conn.close()
        return tables
        
    except Exception as e:
        print(f"Database error: {e}")
        return []

def check_promo_label_structure():
    """
    Check if promo_label has all the features we need
    """
    print("\nPROMO_LABEL TABLE ANALYSIS")
    print("=" * 40)
    
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="casino_research",
            user="researcher",
            password="academic_password_2024"
        )
        
        # Check promo_label structure
        cursor = conn.cursor()
        cursor.execute("""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = 'promo_label' AND table_schema = 'casino_data'
            ORDER BY ordinal_position;
        """)
        
        columns = cursor.fetchall()
        print("PROMO_LABEL COLUMNS:")
        for col_name, col_type in columns:
            print(f"   - {col_name}: {col_type}")
        
        # Sample data
        cursor.execute("""
            SELECT * FROM casino_data.promo_label 
            WHERE period = '2022-H1' 
            LIMIT 5;
        """)
        
        sample_data = cursor.fetchall()
        print(f"\nSAMPLE DATA (2022-H1):")
        col_names = [desc[0] for desc in cursor.description]
        
        for i, row in enumerate(sample_data):
            print(f"   Record {i+1}:")
            for j, value in enumerate(row):
                print(f"     {col_names[j]}: {value}")
        
        # Check label distribution for one period
        cursor.execute("""
            SELECT promo_label, COUNT(*) as count
            FROM casino_data.promo_label 
            WHERE period = '2022-H1'
            GROUP BY promo_label
            ORDER BY count DESC;
        """)
        
        label_dist = cursor.fetchall()
        print(f"\nLABEL DISTRIBUTION (2022-H1):")
        total = sum([count for _, count in label_dist])
        
        for label, count in label_dist:
            pct = (count/total)*100
            print(f"   {label}: {count} ({pct:.1f}%)")
        
        # Calculate imbalance
        if label_dist:
            max_count = max([count for _, count in label_dist])
            min_count = min([count for _, count in label_dist])
            imbalance_ratio = max_count / min_count
            
            print(f"\nIMBALANCE ANALYSIS:")
            print(f"   Ratio: {imbalance_ratio:.1f}:1")
            
            if imbalance_ratio > 10:
                print("   CRITICAL IMBALANCE - This explains single class prediction!")
            elif imbalance_ratio > 5:
                print("   MODERATE IMBALANCE - Needs balancing")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"Error analyzing promo_label: {e}")

def suggest_solutions():
    """
    Suggest solutions based on findings
    """
    print("\nSOLUTION RECOMMENDATIONS")
    print("=" * 40)
    
    print("PROBLEM IDENTIFIED:")
    print("   - customer_features table is missing")
    print("   - Only promo_label table exists")
    print("   - Cannot train RF without features")
    
    print("\nPOSSIBLE SOLUTIONS:")
    
    print("\n1. RECREATE FEATURES TABLE:")
    print("   - Run your feature engineering pipeline")
    print("   - Check if customer_features_robust exists")
    print("   - Look for backup tables")
    
    print("\n2. USE EXISTING CUSTOMER DATA:")
    print("   - Check if customer data exists in other tables")
    print("   - Extract features from raw transaction data")
    print("   - Rebuild feature engineering pipeline")
    
    print("\n3. EMERGENCY SIMULATION:")
    print("   - Create synthetic features based on promo_label")
    print("   - Use for testing RF class balancing")
    print("   - Not ideal for final thesis")
    
    solution_code = '''
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
    
    print(f"\\nPREDICTION RESULTS:")
    print(f"Total classes: {total_classes}")
    print(f"Predicted classes: {predicted_classes}")
    
    for class_idx, count in pred_dist.items():
        class_name = le.inverse_transform([class_idx])[0]
        pct = (count/len(y_pred))*100
        print(f"   {class_name}: {count} ({pct:.1f}%)")
    
    if predicted_classes == total_classes:
        print("\\nSUCCESS: All classes predicted!")
        print("The balancing technique works - now apply to real features")
    else:
        print(f"\\nSTILL MISSING: {total_classes - predicted_classes} classes")
        print("Need more aggressive balancing")

if __name__ == "__main__":
    test_rf_with_emergency_features('2022-H1')
'''
    
    with open('emergency_feature_test.py', 'w') as f:
        f.write(solution_code)
    
    print("\n4. EMERGENCY TEST CODE GENERATED:")
    print("   File: emergency_feature_test.py")
    print("   Use this to test RF balancing while you recreate features")

def main():
    """
    Main analysis function
    """
    print("FIND MISSING CUSTOMER FEATURES TABLES")
    print("=" * 50)
    
    # Find all tables
    tables = find_all_tables()
    
    # Analyze promo_label
    check_promo_label_structure()
    
    # Suggest solutions
    suggest_solutions()
    
    print("\nNEXT ACTIONS:")
    print("1. Check if you have backup of customer_features table")
    print("2. Re-run your feature engineering pipeline")
    print("3. Use emergency_feature_test.py to test RF balancing")
    print("4. Once features are restored, apply the balancing fixes")

if __name__ == "__main__":
    main()