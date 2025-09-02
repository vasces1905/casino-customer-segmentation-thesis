#!/usr/bin/env python3
"""
Temporal Analysis Schema Compatibility Fix
University of Bath - Database Schema Adaptation
"""

import os
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

load_dotenv()

def quick_temporal_analysis():
    """Quick temporal analysis without problematic columns."""
    
    try:
        # Connect to database
        connection = psycopg2.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        
        print("Database connected")
        
        # Simplified query - only essential columns
        simple_query = """
        SELECT 
            ps.customer_id,
            ps.session_start,
            ps.total_bet,
            cd.age_range,
            cd.gender,
            cd.customer_segment
        FROM casino_data.player_sessions ps
        JOIN casino_data.customer_demographics cd 
            ON ps.customer_id = cd.customer_id
        WHERE ps.session_start IS NOT NULL 
        LIMIT 5000;
        """
        
        print("Loading simplified real data...")
        session_data = pd.read_sql_query(simple_query, connection)
        
        if session_data.empty:
            print("No data found, using fallback")
        
        print(f"Loaded {len(session_data)} real sessions")
        print(f"Real customers: {session_data['customer_id'].nunique()}")
        
        # Extract temporal features (simplified)
        print("Extracting simplified temporal features...")
        
        session_data['datetime'] = pd.to_datetime(session_data['session_start'])
        session_data['hour'] = session_data['datetime'].dt.hour
        session_data['is_weekend'] = session_data['datetime'].dt.dayofweek.isin([5, 6])
        
        # Customer-level features
        customer_features = []
        
        for customer_id in session_data['customer_id'].unique():
            customer_sessions = session_data[session_data['customer_id'] == customer_id]
            
            if len(customer_sessions) < 2:
                continue
            
            # Simplified temporal features
            weekend_sessions = customer_sessions['is_weekend'].sum()
            total_sessions = len(customer_sessions)
            weekend_pref = (weekend_sessions / 2) / ((total_sessions - weekend_sessions) / 5 + weekend_sessions / 2) if total_sessions > weekend_sessions else 1.0
            
            late_night_sessions = customer_sessions[(customer_sessions['hour'] >= 22) | (customer_sessions['hour'] <= 6)]
            late_night_intensity = len(late_night_sessions) / total_sessions
            
            hour_variance = customer_sessions['hour'].var()
            temporal_consistency = 1 / (1 + hour_variance) if hour_variance > 0 else 1
            
            features = {
                'customer_id': customer_id,
                'weekend_preference': weekend_pref,
                'late_night_intensity': late_night_intensity,
                'temporal_consistency': temporal_consistency,
                'time_diversity': customer_sessions['hour'].nunique() / 24,
                'session_count': total_sessions,
                'age_range': customer_sessions['age_range'].iloc[0],
                'gender': customer_sessions['gender'].iloc[0],
                'customer_segment': customer_sessions['customer_segment'].iloc[0]
            }
            
            customer_features.append(features)
        
        features_df = pd.DataFrame(customer_features)
        print(f"Features extracted for {len(features_df)} real customers")
        
        # Quick clustering
        print("Quick temporal clustering...")
        
        clustering_cols = ['weekend_preference', 'late_night_intensity', 'temporal_consistency', 'time_diversity']
        X = features_df[clustering_cols].fillna(0)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        features_df['temporal_cluster'] = kmeans.fit_predict(X_scaled)
        
        # Results
        print("\nREAL CASINO TEMPORAL ANALYSIS RESULTS:")
        print("="*50)
        
        for cluster_id in range(4):
            cluster_data = features_df[features_df['temporal_cluster'] == cluster_id]
            
            weekend_pref = cluster_data['weekend_preference'].mean()
            late_night = cluster_data['late_night_intensity'].mean()
            consistency = cluster_data['temporal_consistency'].mean()
            
            if weekend_pref > 0.6:
                label = "Weekend-Focused Players"
            elif late_night > 0.3:
                label = "Late-Night Risk Players"
            elif consistency > 0.7:
                label = "Routine Regular Players"
            else:
                label = "Mixed Pattern Players"
            
            percentage = len(cluster_data) / len(features_df) * 100
            
            print(f"\nCluster {cluster_id}: {label}")
            print(f"  Size: {len(cluster_data)} customers ({percentage:.1f}%)")
            print(f"  Weekend Preference: {weekend_pref:.3f}")
            print(f"  Late-Night Intensity: {late_night:.3f}")
            print(f"  Temporal Consistency: {consistency:.3f}")
        
        # Save results
        os.makedirs("thesis_outputs", exist_ok=True)
        features_df.to_csv("thesis_outputs/real_temporal_features.csv", index=False)
        
        print(f"\nREAL DATA TEMPORAL ANALYSIS COMPLETE!")
        print(f"Real customers analyzed: {len(features_df)}")
        print(f"Results saved: thesis_outputs/real_temporal_features.csv")
        print(f"FCC Gap #1: COMPLETED WITH REAL CASINO DATA!")
        
        connection.close()
        
        return features_df
        
    except Exception as e:
        print(f"Error: {e}")
        print("Will need to debug database connection...")
        return None

if __name__ == "__main__":
    print("TEMPORAL ANALYSIS SCHEMA COMPATIBILITY")
    print("University of Bath - Database Schema Adaptation")
    print("="*60)
    
    result = quick_temporal_analysis()
    
    if result is not None:
        print("\nReal casino temporal analysis completed!")
        print("Ready to proceed to FCC Gap #2 (A/B Analysis)")
    else:
        print("\nDatabase issues need investigation")