#!/usr/bin/env python3
"""
Corrected Pipeline Diagnosis with Proper Table Names
"""

import pandas as pd
import numpy as np
import psycopg2
import joblib
from collections import Counter
import os

def diagnose_prediction_issue_corrected(period='2022-H1'):
    """
    Corrected diagnosis function with proper table names
    """
    print(f"PREDICTION DIAGNOSIS - {period}")
    print("=" * 50)
    
    conn = psycopg2.connect(
        host="localhost",
        database="casino_research",
        user="postgres",  # Update with your credentials
        password="your_password"
    )
    
    # Try different table name variations
    possible_queries = [
        # Option 1: customer_features_robust
        f"""
        SELECT cfr.*, pl.promo_label
        FROM customer_features_robust cfr
        JOIN promo_label pl ON cfr.customer_id = pl.customer_id 
        WHERE pl.period = '{period}'
        LIMIT 1000;
        """,
        
        # Option 2: customer_features (if it exists)
        f"""
        SELECT cf.*, pl.promo_label
        FROM customer_features cf
        JOIN promo_label pl ON cf.customer_id = pl.customer_id 
        WHERE pl.period = '{period}'
        LIMIT 1000;
        """,
        
        # Option 3: Just promo_label table
        f"""
        SELECT customer_id, promo_label, period
        FROM casino_data.promo_label 
        WHERE period = '{period}'
        LIMIT 1000;
        """
    ]
    
    df = None
    for i, query in enumerate(possible_queries):
        try:
            print(f"Trying query option {i+1}...")
            df = pd.read_sql(query, conn)
            print(f"Success! Loaded {len(df)} records")
            break
        except Exception as e:
            print(f"Query {i+1} failed: {e}")
            continue
    
    if df is None:
        print("All queries failed. Check table names.")
        conn.close()
        return
    
    # Analyze label distribution
    if 'promo_label' in df.columns:
        label_dist = Counter(df['promo_label'])
        print(f"ACTUAL LABEL DISTRIBUTION:")
        total = len(df)
        for label, count in label_dist.items():
            pct = (count/total)*100
            print(f"   {label}: {count} ({pct:.1f}%)")
        
        # Calculate imbalance ratio
        max_count = max(label_dist.values())
        min_count = min(label_dist.values())
        imbalance_ratio = max_count / min_count
        
        print(f"\nIBALANCE ANALYSIS:")
        print(f"   Dominant class: {max_count} samples")
        print(f"   Minority class: {min_count} samples")
        print(f"   Imbalance ratio: {imbalance_ratio:.1f}:1")
        
        if imbalance_ratio > 10:
            print("   CRITICAL IMBALANCE DETECTED")
            print("   This explains why RF predicts only 1-2 classes")
        elif imbalance_ratio > 5:
            print("   MODERATE IMBALANCE - Needs balancing")
        else:
            print("   BALANCED - Look for other issues")
    
    conn.close()
    return df

def check_existing_models():
    """
    Check for existing model files
    """
    print("CHECKING EXISTING MODELS")
    print("=" * 30)
    
    model_dirs = ["models", "models/active", "./"]
    
    for directory in model_dirs:
        if os.path.exists(directory):
            pkl_files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
            if pkl_files:
                print(f"Found PKL files in {directory}:")
                for pkl in pkl_files:
                    print(f"   - {pkl}")
                return directory, pkl_files
    
    print("No PKL files found in standard directories")
    return None, []

def main():
    """
    Main corrected diagnosis function
    """
    print("CORRECTED PIPELINE DIAGNOSIS")
    print("=" * 50)
    
    # Create directories
    create_missing_directories()
    
    # Check database structure
    feature_tables = check_database_structure()
    
    # Check existing models
    model_dir, pkl_files = check_existing_models()
    
    # Diagnose each period
    periods = ['2022-H1', '2022-H2', '2023-H1', '2023-H2']
    for period in periods:
        try:
            df = diagnose_prediction_issue_corrected(period)
            if df is not None:
                print(f"Period {period}: Analysis completed")
            else:
                print(f"Period {period}: Analysis failed")
        except Exception as e:
            print(f"Error in {period}: {e}")

if __name__ == "__main__":
    main()
