#!/usr/bin/env python3
"""
Fix 2023-H2 JSON Error - Single Period Re-run
============================================
Quick fix for JSON serialization error in 2023-H2
"""

import numpy as np
import pandas as pd
import psycopg2
import json
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    return obj

def fix_2023_h2_segmentation():
    """Fix 2023-H2 segmentation with proper JSON handling"""
    
    # Database connection
    conn = psycopg2.connect(
        host="localhost",
        database="casino_research",
        user="researcher",
        password="academic_password_2024"
    )
    
    logger.info("🔧 Loading 2023-H2 clean data...")
    
    # Load clean data for 2023-H2
    query = """
    SELECT 
        customer_id,
        total_bet,
        avg_bet,
        loss_rate,
        COALESCE(bet_volatility, 0) as bet_volatility,
        total_sessions,
        COALESCE(loss_chasing_score, 0) as loss_chasing_score
    FROM casino_data.customer_features
    WHERE analysis_period = '2023-H2'
        AND total_bet > 0
        AND total_bet <= 1500000
        AND avg_bet > 0
    ORDER BY customer_id;
    """
    
    df = pd.read_sql_query(query, conn)
    logger.info(f"✅ Loaded {len(df)} customers for 2023-H2")
    
    # Prepare features
    feature_columns = ['total_bet', 'avg_bet', 'loss_rate', 'bet_volatility', 'total_sessions', 'loss_chasing_score']
    X = df[feature_columns].fillna(df[feature_columns].median())
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # KMeans clustering
    logger.info("🔄 Running KMeans clustering...")
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculate metrics
    silhouette = silhouette_score(X_scaled, labels)
    davies_bouldin = davies_bouldin_score(X_scaled, labels)
    
    logger.info(f"✅ Clustering completed: Silhouette {silhouette:.4f}")
    
    # Create segment profiles
    df_temp = df.copy()
    df_temp['cluster'] = labels
    
    segment_profiles = {}
    for cluster_id in range(4):
        cluster_data = df_temp[df_temp['cluster'] == cluster_id]
        
        if len(cluster_data) == 0:
            continue
        
        # Safe avg_session_value calculation
        try:
            total_sessions_safe = cluster_data['total_sessions'].replace(0, 1)
            avg_session_value = (cluster_data['total_bet'] / total_sessions_safe).mean()
            if np.isnan(avg_session_value) or np.isinf(avg_session_value):
                avg_session_value = 0.0
        except:
            avg_session_value = 0.0
        
        profile = {
            'cluster_id': cluster_id,
            'size': len(cluster_data),
            'percentage': round(len(cluster_data) / len(df_temp) * 100, 2),
            'avg_total_bet': round(cluster_data['total_bet'].mean(), 2),
            'avg_loss_rate': round(cluster_data['loss_rate'].mean(), 2),
            'avg_session_value': round(avg_session_value, 2),
            'avg_sessions': round(cluster_data['total_sessions'].mean(), 2),
            'avg_loss_chasing': round(cluster_data['loss_chasing_score'].mean(), 2)
        }
        
        # Business label assignment
        if profile['avg_total_bet'] > 5000 and profile['avg_sessions'] > 10:
            profile['business_label'] = "High_Value_Player"
        elif profile['avg_loss_chasing'] > 30 or profile['avg_loss_rate'] > 20:
            profile['business_label'] = "At_Risk_Player"
        elif profile['avg_sessions'] > 3 and profile['avg_total_bet'] > 800:
            profile['business_label'] = "Regular_Player"
        else:
            profile['business_label'] = "Casual_Player"
        
        segment_profiles[cluster_id] = profile
    
    # Save to database with proper JSON conversion
    logger.info("💾 Saving to database...")
    cursor = conn.cursor()
    
    # Clear existing 2023-H2 data
    cursor.execute("DELETE FROM casino_data.kmeans_segments WHERE period_id = '2023-H2'")
    
    # Insert new data
    for i, customer_id in enumerate(df['customer_id']):
        cluster_id = int(labels[i])
        cluster_profile = segment_profiles.get(cluster_id, {})
        cluster_label = cluster_profile.get('business_label', f'Cluster_{cluster_id}')
        
        # Create metadata (convert all numpy types)
        metadata = convert_numpy_types({
            'algorithm': 'kmeans_clean',
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies_bouldin,
            'data_quality': 'A_GRADE_CLEAN',
            'max_bet_cap': 1500000,
            'academic_approval': 'Bath University A Excellent',
            'timestamp': datetime.now().isoformat()
        })
        
        # Convert segment profile
        segment_data = convert_numpy_types(cluster_profile)
        
        cursor.execute("""
            INSERT INTO casino_data.kmeans_segments (
                customer_id, period_id, cluster_id, cluster_label,
                silhouette_score, distance_to_centroid, model_metadata,
                kmeans_version, segment_data, avg_session_from_metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            int(customer_id),
            '2023-H2',
            cluster_id,
            cluster_label,
            float(silhouette),
            0.0,
            json.dumps(metadata),
            2,  # Clean version
            json.dumps(segment_data),
            float(cluster_profile.get('avg_session_value', 0.0))
        ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logger.info("✅ 2023-H2 segmentation fixed and saved!")
    
    # Print segment summary
    print("\n🎯 2023-H2 SEGMENT SUMMARY:")
    print("="*50)
    for cluster_id, profile in segment_profiles.items():
        print(f"{profile['business_label']}: {profile['size']} customers ({profile['percentage']}%)")
        print(f"  Avg Bet: €{profile['avg_total_bet']}, Avg Sessions: {profile['avg_sessions']}")
    
    print(f"\n✅ 2023-H2 SUCCESSFULLY FIXED!")
    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Total Customers: {len(df)}")

if __name__ == "__main__":
    fix_2023_h2_segmentation()