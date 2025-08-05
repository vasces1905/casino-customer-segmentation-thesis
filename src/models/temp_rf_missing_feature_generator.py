"""
Missing Feature Generator - Step 3
Generate the missing features that models expect
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import psycopg2
from typing import Dict, Any

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'casino_research',
    'user': 'researcher',
    'password': 'academic_password_2024'
}

def load_customer_data(period: str = '2023-H2') -> pd.DataFrame:
    """Load customer data from database."""
    query = """
    SELECT * FROM casino_data.customer_features 
    WHERE analysis_period = %s
    ORDER BY customer_id
    """
    
    conn = psycopg2.connect(**DB_CONFIG)
    df = pd.read_sql_query(query, conn, params=(period,))
    conn.close()
    
    print(f"✅ Loaded {len(df)} customers from period {period}")
    return df

def generate_kmeans_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate K-means clustering features."""
    print("🔄 Generating K-means features...")
    
    # Features for clustering (behavioral metrics)
    clustering_features = [
        'total_bet', 'avg_bet', 'bet_volatility', 'total_sessions',
        'loss_rate', 'loss_chasing_score', 'game_diversity'
    ]
    
    # Ensure all features exist
    available_features = [f for f in clustering_features if f in df.columns]
    X_cluster = df[available_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Apply K-means clustering (try different cluster numbers)
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Create K-means encoded feature
    df['kmeans_segment_encoded'] = cluster_labels
    
    print(f"✅ K-means clustering complete - {len(np.unique(cluster_labels))} clusters")
    return df

def generate_dbscan_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate DBSCAN outlier detection features."""
    print("🔄 Generating DBSCAN outlier features...")
    
    # Features for outlier detection
    outlier_features = [
        'total_bet', 'avg_bet', 'bet_volatility', 'loss_rate'
    ]
    
    available_features = [f for f in outlier_features if f in df.columns]
    X_outlier = df[available_features].fillna(0)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_outlier)
    
    # Apply DBSCAN
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Create DBSCAN outlier indicator (outliers have label -1)
    df['is_dbscan_outlier'] = (dbscan_labels == -1).astype(int)
    
    outlier_count = df['is_dbscan_outlier'].sum()
    print(f"✅ DBSCAN complete - {outlier_count} outliers detected")
    return df

def generate_value_tier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate value tier and high value features."""
    print("🔄 Generating value tier features...")
    
    # High value indicator (top 10% by total_bet)
    high_value_threshold = df['total_bet'].quantile(0.90)
    df['is_high_value'] = (df['total_bet'] >= high_value_threshold).astype(int)
    
    # Personal vs segment ratio (simplified)
    # Use loss_rate as proxy for personal performance vs segment average
    segment_avg_loss = df.groupby('kmeans_segment_encoded')['loss_rate'].transform('mean')
    df['personal_vs_segment_ratio'] = df['loss_rate'] / (segment_avg_loss + 0.001)  # Avoid division by zero
    
    print("✅ Value tier features generated")
    return df

def generate_session_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate session-based features."""
    print("🔄 Generating session features...")
    
    # Segment average session (based on clustering)
    df['segment_avg_session'] = df.groupby('kmeans_segment_encoded')['total_sessions'].transform('mean')
    
    print("✅ Session features generated")
    return df

def generate_intervention_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate intervention need features."""
    print("🔄 Generating intervention features...")
    
    # Needs intervention based on high loss chasing score and high loss rate
    high_risk_threshold = df['loss_chasing_score'].quantile(0.80)
    high_loss_threshold = df['loss_rate'].quantile(0.75)
    
    df['needs_intervention'] = (
        (df['loss_chasing_score'] >= high_risk_threshold) & 
        (df['loss_rate'] >= high_loss_threshold)
    ).astype(int)
    
    intervention_count = df['needs_intervention'].sum()
    print(f"✅ Intervention features generated - {intervention_count} customers need intervention")
    return df

def generate_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate interaction features."""
    print("🔄 Generating interaction features...")
    
    # Outlier risk interaction
    df['outlier_risk_interaction'] = (
        df['is_dbscan_outlier'] * df['loss_chasing_score']
    )
    
    print("✅ Interaction features generated")
    return df

def generate_all_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate all missing features that models expect."""
    
    print("🚀 GENERATING ALL MISSING FEATURES")
    print("=" * 50)
    
    # Generate features step by step
    df = generate_kmeans_features(df)
    df = generate_dbscan_features(df)
    df = generate_value_tier_features(df)
    df = generate_session_features(df)
    df = generate_intervention_features(df)
    df = generate_interaction_features(df)
    
    # Also add basic engineered features
    df['total_bet_original'] = df['total_bet']
    df['avg_bet_original'] = df['avg_bet']
    df['bet_volatility_original'] = df['bet_volatility']
    df['total_sessions_original'] = df['total_sessions']
    
    # Log transformations
    df['log_total_bet'] = np.log1p(df['total_bet'])
    df['log_avg_bet'] = np.log1p(df['avg_bet'])
    df['log_bet_volatility'] = np.log1p(df['bet_volatility'])
    df['log_total_sessions'] = np.log1p(df['total_sessions'])
    
    # Robust versions
    df['total_bet_robust'] = np.minimum(df['total_bet'], df['total_bet'].quantile(0.95))
    df['avg_bet_robust'] = np.minimum(df['avg_bet'], df['avg_bet'].quantile(0.95))
    df['bet_volatility_robust'] = np.minimum(df['bet_volatility'], df['bet_volatility'].quantile(0.95))
    
    # Binary indicators
    df['is_ultra_high_roller'] = (df['total_bet'] > df['total_bet'].quantile(0.99)).astype(int)
    df['is_millionaire'] = (df['total_bet'] > 1000000).astype(int)
    df['is_high_avg_better'] = (df['avg_bet'] > df['avg_bet'].quantile(0.90)).astype(int)
    
    print(f"\n✅ ALL FEATURES GENERATED")
    print(f"   Original features: {len([c for c in df.columns if not any(x in c for x in ['kmeans', 'dbscan', 'is_', 'log_', 'robust', 'original', 'interaction', 'intervention'])])}")
    print(f"   Generated features: {len([c for c in df.columns if any(x in c for x in ['kmeans', 'dbscan', 'is_', 'log_', 'robust', 'original', 'interaction', 'intervention'])])}")
    print(f"   Total features: {len(df.columns)}")
    
    return df

def save_enhanced_data(df: pd.DataFrame, period: str = '2023-H2') -> str:
    """Save enhanced data to CSV file."""
    filename = f'enhanced_customer_features_{period}_{pd.Timestamp.now().strftime("%Y%m%d_%H%M")}.csv'
    df.to_csv(filename, index=False)
    print(f"💾 Enhanced data saved to: {filename}")
    return filename

def main():
    """Generate missing features for ensemble prediction."""
    
    try:
        # Load customer data
        print("📊 LOADING CUSTOMER DATA...")
        df = load_customer_data('2023-H2')
        
        # Generate all missing features
        df_enhanced = generate_all_missing_features(df)
        
        # Save enhanced data
        filename = save_enhanced_data(df_enhanced, '2023-H2')
        
        # Print feature summary
        print(f"\n📋 FEATURE SUMMARY:")
        
        # Missing features that were generated
        missing_features = [
            'outlier_risk_interaction', 'is_high_value', 'personal_vs_segment_ratio',
            'segment_avg_session', 'needs_intervention', 'is_dbscan_outlier', 'kmeans_segment_encoded'
        ]
        
        for feature in missing_features:
            if feature in df_enhanced.columns:
                print(f"   ✅ {feature}")
            else:
                print(f"   ❌ {feature} - STILL MISSING")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"1. Use the enhanced dataset: {filename}")
        print(f"2. Update ensemble script to load this file instead of database")
        print(f"3. Run ensemble prediction again")
        
        return df_enhanced, filename
        
    except Exception as e:
        print(f"❌ Feature generation failed: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    enhanced_data, filename = main()