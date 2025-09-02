"""Temporal Promotion Evolution Analysis for 2024

University of Bath - MSc Computer Science
Student: Muhammed Yavuzhan CANLI
Ethics Approval: 10351-12382

Generates 2024-H1 promotion predictions for temporal evolution visualization.
Integrates Random Forest predictions with customer segmentation for comprehensive analysis.
"""

import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from datetime import datetime

def generate_2024_temporal_evolution():
    """Generate predictions for 2024-H1 and create CSV for visualization"""
    
    print("Generating 2024-H1 predictions for temporal visualization")
    
    # Database connection
    engine = create_engine("postgresql://researcher:academic_password_2024@localhost:5432/casino_research")
    
    # Load 2024-H1 model
    model_path = "src/models/models/generic_rf/clean_harmonized_rf_2024-H1_v1_20250818_1653.pkl"
    model_dict = joblib.load(model_path)
    
    rf_model = model_dict['model']
    scaler = model_dict['scaler']
    features = model_dict['features']
    
    # Load 2024-H1 data
    query = """
    SELECT 
        cf.customer_id,
        cf.total_bet,
        cf.avg_bet,
        cf.loss_rate,
        cf.loss_chasing_score,
        ks.cluster_id
    FROM casino_data.customer_features cf
    JOIN casino_data.kmeans_segments ks 
        ON cf.customer_id = ks.customer_id
    WHERE cf.analysis_period = '2024-H1' 
        AND ks.period_id = '2024-H1'
    """
    
    data = pd.read_sql(query, engine)
    print(f"Loaded {len(data)} customers for 2024-H1")
    
    # Prepare features
    X = data[['total_bet', 'avg_bet', 'loss_rate', 'loss_chasing_score']].fillna(0)
    X_scaled = scaler.transform(X)
    
    # Make predictions
    predictions = rf_model.predict(X_scaled)
    probabilities = rf_model.predict_proba(X_scaled)
    confidence_scores = probabilities.max(axis=1)
    
    # Map cluster IDs to promotion labels
    promotion_mapping = {
        0: 'NO_PROMOTION',
        1: 'GROWTH_TARGET', 
        2: 'LOW_ENGAGEMENT',
        3: 'INTERVENTION_NEEDED'
    }
    
    # Map to business priority
    def get_priority(confidence, promotion):
        if promotion == 'INTERVENTION_NEEDED':
            return 'URGENT'
        elif promotion == 'NO_PROMOTION':
            return 'MAINTAIN'
        elif confidence > 0.8:
            return 'HIGH'
        elif confidence > 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    # Create predictions dataframe
    predictions_df = pd.DataFrame({
        'customer_id': data['customer_id'],
        'period': '2024-H1',
        'predicted_promotion': [promotion_mapping[pred] for pred in predictions],
        'confidence': confidence_scores.round(4),
        'business_priority': [get_priority(conf, promotion_mapping[pred]) 
                             for conf, pred in zip(confidence_scores, predictions)],
        'customer_status': 'NEW_CUSTOMER'
    })
    
    # Save to CSV
    output_file = f'unified_temporal_evolution_2024_H1_{datetime.now().strftime("%Y%m%d")}.csv'
    predictions_df.to_csv(output_file, index=False)
    
    print(f"\nCSV created: {output_file}")
    print(f"Total predictions: {len(predictions_df)}")
    print("\nPromotion distribution:")
    print(predictions_df['predicted_promotion'].value_counts())
    print("\nPriority distribution:")
    print(predictions_df['business_priority'].value_counts())
    
    # Integrate with existing temporal data if available
    try:
        existing_data = pd.read_csv('unified_temporal_evolution_20250818.csv')
        combined_data = pd.concat([existing_data, predictions_df], ignore_index=True)
        combined_file = f'unified_temporal_evolution_complete_{datetime.now().strftime("%Y%m%d")}.csv'
        combined_data.to_csv(combined_file, index=False)
        print(f"\nCombined CSV created: {combined_file}")
    except FileNotFoundError:
        print("\nNo existing data found, only 2024-H1 saved")
    
    return predictions_df

if __name__ == "__main__":
    df = generate_2024_temporal_evolution()
    
    # Now run visualization
    print("\nReady to run temporal_promotion_evolution_system.py for visualizations!")