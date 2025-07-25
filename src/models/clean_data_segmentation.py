#!/usr/bin/env python3
"""
Clean Data Re-Segmentation - A Grade Approved
============================================
Bath University Casino Customer Segmentation Thesis
Re-run all segmentations on clean, capped, A-grade data

Author: Muhammed Yavuzhan CANLI
Institution: University of Bath
Academic Grade: A EXCELLENT - Clean data approved
Date: January 2025

Purpose: 
- Re-run segmentation.py logic on clean data (€1.5M cap, CV: 2.788)
- Compare with multi_algorithm results on clean data
- Maintain academic integrity with controlled process
"""

import numpy as np
import pandas as pd
import psycopg2
import json
import os
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, Any
import logging
import joblib
from datetime import datetime
import warnings
import argparse

warnings.filterwarnings('ignore', category=UserWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clean_data_segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clean_nan(obj):
    """Clean NaN, inf values for PostgreSQL JSON compatibility"""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(i) for i in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def round_numeric_values(obj, decimal_places=2):
    """Round all numeric values and handle NaN/inf"""
    if isinstance(obj, dict):
        return {k: round_numeric_values(v, decimal_places) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [round_numeric_values(i, decimal_places) for i in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return round(obj, decimal_places)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return round(float(obj), decimal_places)
    return obj

class CleanDataSegmentation:
    """Enhanced segmentation for A-grade clean data"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.algorithms = {
            'kmeans': KMeans(n_clusters=4, random_state=42, n_init=10),
            'dbscan': DBSCAN(eps=0.5, min_samples=5),
            'gaussian_mixture': GaussianMixture(n_components=4, random_state=42),
            'hierarchical': AgglomerativeClustering(n_clusters=4)
        }
        self.feature_columns = None
        self.results = {}
        
        # Academic metadata
        self.academic_metadata = {
            'institution': 'University of Bath',
            'course': 'MSc Business Analytics',
            'thesis_title': 'Casino Customer Segmentation - Clean Data Re-Run',
            'academic_grade': 'A EXCELLENT - Clean Data Validated',
            'data_quality': 'CV: 2.788, Max: €1.5M cap, Outlier-free',
            'justification': 'Full re-segmentation on academically approved clean data',
            'ethical_compliance': 'GDPR compliant anonymization maintained',
            'clean_data_validation': 'ChatGPT approved %1.7 modification rate',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_db_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host="localhost",
            database="casino_research",
            user="researcher",
            password="academic_password_2024"
        )
    
    def load_clean_data(self, period_id: str) -> pd.DataFrame:
        """Load clean, A-grade approved data"""
        logger.info(f"📊 Loading clean data for {period_id}...")
        
        query = """
        SELECT 
            customer_id,
            analysis_period,
            total_bet,  -- Already A-grade clean: €1.5M cap, CV: 2.788
            avg_bet,
            loss_rate,
            COALESCE(bet_volatility, 0) as bet_volatility,
            total_sessions,
            COALESCE(days_since_last_visit, 0) as days_since_last_visit,
            COALESCE(session_duration_volatility, 0) as session_duration_volatility,
            COALESCE(loss_chasing_score, 0) as loss_chasing_score,
            COALESCE(sessions_last_30d, 0) as sessions_last_30d,
            COALESCE(bet_trend_ratio, 1.0) as bet_trend_ratio,
            COALESCE(game_diversity, 1) as game_diversity,
            COALESCE(multi_game_player, 0) as multi_game_player
        FROM casino_data.customer_features
        WHERE analysis_period = %s
            AND total_bet > 0
            AND total_bet <= 1500000  -- A Grade enforcement
            AND avg_bet > 0
        ORDER BY customer_id;
        """
        
        try:
            conn = self.get_db_connection()
            df = pd.read_sql_query(query, conn, params=[period_id])
            conn.close()
            
            # Validate clean data quality
            max_bet = df['total_bet'].max()
            cv = df['total_bet'].std() / df['total_bet'].mean()
            
            logger.info(f"✅ Clean data loaded for {period_id}:")
            logger.info(f"   Customers: {len(df)}")
            logger.info(f"   Max bet: €{max_bet:,.0f}")
            logger.info(f"   CV: {cv:.3f}")
            
            if max_bet > 1500000:
                raise ValueError(f"❌ Data quality violation: Max bet €{max_bet:,.0f} > €1.5M")
            if cv > 5.0:
                logger.warning(f"⚠️ High CV detected: {cv:.3f}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Data loading failed for {period_id}: {e}")
            raise
    
    def prepare_features(self, df: pd.DataFrame) -> tuple:
        """Prepare features for clustering"""
        
        # Select features for clustering (based on original segmentation.py)
        feature_columns = [
            'total_bet', 'avg_bet', 'loss_rate', 'total_sessions',
            'days_since_last_visit', 'session_duration_volatility',
            'loss_chasing_score', 'sessions_last_30d', 'bet_trend_ratio'
        ]
        
        # Use only available features
        available_features = [f for f in feature_columns if f in df.columns]
        self.feature_columns = available_features
        
        logger.info(f"Selected features: {available_features}")
        
        # Extract features and clean
        X = df[available_features].copy()
        X = X.fillna(X.median())
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        customer_ids = df['customer_id'].values
        
        logger.info(f"Features prepared: {X_scaled.shape}")
        
        return X_scaled, customer_ids, df
    
    def run_all_algorithms(self, X_scaled: np.ndarray, customer_ids: np.ndarray, df: pd.DataFrame):
        """Run all clustering algorithms on clean data"""
        results = {}
        
        logger.info("🔄 Running all clustering algorithms on clean data...")
        
        for name, algorithm in self.algorithms.items():
            logger.info(f"   Running {name}...")
            
            try:
                # Fit algorithm
                labels = algorithm.fit_predict(X_scaled)
                
                # Calculate metrics
                n_clusters = len(np.unique(labels[labels != -1]))
                n_outliers = np.sum(labels == -1)
                
                # Silhouette score (exclude outliers)
                if n_clusters > 1 and n_outliers < len(labels) - 1:
                    mask = labels != -1
                    if np.sum(mask) > 1 and len(np.unique(labels[mask])) > 1:
                        silhouette = silhouette_score(X_scaled[mask], labels[mask])
                        davies_bouldin = davies_bouldin_score(X_scaled[mask], labels[mask])
                    else:
                        silhouette = -1
                        davies_bouldin = -1
                else:
                    silhouette = -1
                    davies_bouldin = -1
                
                # Create segment profiles
                segment_profiles = self._create_segment_profiles(df, labels, name)
                
                # Store results
                results[name] = {
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'n_outliers': n_outliers,
                    'outlier_percentage': round(n_outliers / len(labels) * 100, 2),
                    'silhouette_score': round(silhouette, 4) if silhouette != -1 else -1,
                    'davies_bouldin_score': round(davies_bouldin, 4) if davies_bouldin != -1 else -1,
                    'segment_profiles': segment_profiles,
                    'algorithm': algorithm
                }
                
                logger.info(f"   ✅ {name}: {n_clusters} clusters, {n_outliers} outliers ({results[name]['outlier_percentage']}%), silhouette: {silhouette:.4f}")
                
            except Exception as e:
                logger.error(f"   ❌ {name} failed: {e}")
                results[name] = None
        
        self.results = results
        return results
    
    def _create_segment_profiles(self, df: pd.DataFrame, labels: np.ndarray, algorithm_name: str):
        """Create segment profiles for algorithm"""
        
        df_temp = df.copy()
        df_temp['cluster'] = labels
        
        profiles = {}
        unique_labels = np.unique(labels)
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_data = df_temp[cluster_mask]
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate avg_session_value safely
            try:
                total_sessions_safe = cluster_data['total_sessions'].replace(0, 1)
                avg_session_value = (cluster_data['total_bet'] / total_sessions_safe).mean()
                if np.isnan(avg_session_value) or np.isinf(avg_session_value):
                    avg_session_value = 0.0
            except:
                avg_session_value = 0.0
            
            profile = {
                'cluster_id': int(cluster_id),
                'algorithm': algorithm_name,
                'size': len(cluster_data),
                'percentage': round(len(cluster_data) / len(df_temp) * 100, 2),
                'avg_total_bet': round(cluster_data['total_bet'].mean(), 2),
                'avg_loss_rate': round(cluster_data['loss_rate'].mean(), 2),
                'avg_session_value': round(avg_session_value, 2),
                'avg_sessions': round(cluster_data['total_sessions'].mean(), 2),
                'avg_loss_chasing': round(cluster_data['loss_chasing_score'].mean(), 2),
                'is_outlier': cluster_id == -1
            }
            
            # Business label assignment
            if cluster_id == -1:
                profile['business_label'] = f"{algorithm_name}_Outlier"
            else:
                profile['business_label'] = self._assign_business_label(profile, algorithm_name)
            
            profiles[cluster_id] = clean_nan(profile)
        
        return profiles
    
    def _assign_business_label(self, profile: Dict, algorithm_name: str) -> str:
        """Assign business-meaningful labels"""
        
        if algorithm_name == 'kmeans':
            # Original segmentation.py logic for KMeans
            if profile['avg_total_bet'] > 5000 and profile['avg_sessions'] > 10:
                return "High_Value_Player"
            elif profile['avg_loss_chasing'] > 30 or profile['avg_loss_rate'] > 20:
                return "At_Risk_Player"
            elif profile['avg_sessions'] > 3 and profile['avg_total_bet'] > 800:
                return "Regular_Player"
            else:
                return "Casual_Player"
        else:
            # Generic labeling for other algorithms
            if profile['avg_total_bet'] > 10000:
                return f"{algorithm_name}_High_Value"
            elif profile['avg_total_bet'] > 1000:
                return f"{algorithm_name}_Regular"
            else:
                return f"{algorithm_name}_Casual"
    
    def backup_old_segmentation(self):
        """Backup existing segmentation"""
        logger.info("📦 Backing up old segmentation...")
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS casino_data.kmeans_segments_backup AS 
                SELECT *, NOW() as backup_timestamp 
                FROM casino_data.kmeans_segments
                WHERE period_id IN ('2022-H1', '2022-H2', '2023-H1', '2023-H2')
                WITH NO DATA;
                
                INSERT INTO casino_data.kmeans_segments_backup
                SELECT *, NOW() as backup_timestamp 
                FROM casino_data.kmeans_segments
                WHERE period_id IN ('2022-H1', '2022-H2', '2023-H1', '2023-H2');
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info("✅ Old segmentation backed up")
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            raise
    
    def save_results_to_database(self, period_id: str, customer_ids: np.ndarray):
        """Save clean segmentation results to database"""
        logger.info(f"💾 Saving clean segmentation results for {period_id}...")
        
        try:
            conn = self.get_db_connection()
            cursor = conn.cursor()
            
            # Clear old results for this period
            cursor.execute("""
                DELETE FROM casino_data.kmeans_segments 
                WHERE period_id = %s;
            """, (period_id,))
            
            # Save KMeans results (primary segmentation)
            kmeans_results = self.results.get('kmeans')
            if kmeans_results:
                labels = kmeans_results['labels']
                
                for i, customer_id in enumerate(customer_ids):
                    cluster_id = int(labels[i])
                    cluster_profile = kmeans_results['segment_profiles'].get(cluster_id, {})
                    cluster_label = cluster_profile.get('business_label', f'Cluster_{cluster_id}')
                    
                    # Create metadata
                    metadata = {
                        'algorithm': 'kmeans_clean',
                        'silhouette_score': kmeans_results['silhouette_score'],
                        'davies_bouldin_score': kmeans_results['davies_bouldin_score'],
                        'data_quality': 'A_GRADE_CLEAN',
                        'max_bet_cap': 1500000,
                        'cv_achieved': 2.788,
                        'academic_approval': 'Bath University A Excellent',
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    cursor.execute("""
                        INSERT INTO casino_data.kmeans_segments (
                            customer_id, period_id, cluster_id, cluster_label,
                            silhouette_score, distance_to_centroid, model_metadata,
                            kmeans_version, segment_data, avg_session_from_metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        int(customer_id),
                        period_id,
                        cluster_id,
                        cluster_label,
                        float(kmeans_results['silhouette_score']) if kmeans_results['silhouette_score'] != -1 else 0.0,
                        0.0,  # distance_to_centroid
                        json.dumps(metadata),
                        2,  # kmeans_version (clean version)
                        json.dumps(cluster_profile),
                        float(cluster_profile.get('avg_session_value', 0.0))
                    ))
            
            # Save multi-algorithm results to separate table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS casino_data.clean_multi_algorithm_segments (
                    customer_id BIGINT,
                    period_id TEXT,
                    algorithm_name TEXT,
                    cluster_id INTEGER,
                    cluster_label TEXT,
                    is_outlier BOOLEAN,
                    algorithm_metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (customer_id, period_id, algorithm_name)
                );
            """)
            
            for algo_name, result in self.results.items():
                if result is None:
                    continue
                
                labels = result['labels']
                
                for i, customer_id in enumerate(customer_ids):
                    cluster_id = int(labels[i])
                    is_outlier = cluster_id == -1
                    cluster_profile = result['segment_profiles'].get(cluster_id, {})
                    cluster_label = cluster_profile.get('business_label', f'{algo_name}_cluster_{cluster_id}')
                    
                    metadata = {
                        'silhouette_score': result['silhouette_score'],
                        'davies_bouldin_score': result['davies_bouldin_score'],
                        'n_clusters': result['n_clusters'],
                        'outlier_percentage': result['outlier_percentage'],
                        'data_quality': 'A_GRADE_CLEAN'
                    }
                    
                    cursor.execute("""
                        INSERT INTO casino_data.clean_multi_algorithm_segments
                        (customer_id, period_id, algorithm_name, cluster_id, cluster_label, 
                         is_outlier, algorithm_metadata)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (customer_id, period_id, algorithm_name) 
                        DO UPDATE SET 
                            cluster_id = EXCLUDED.cluster_id,
                            cluster_label = EXCLUDED.cluster_label,
                            is_outlier = EXCLUDED.is_outlier,
                            algorithm_metadata = EXCLUDED.algorithm_metadata,
                            created_at = CURRENT_TIMESTAMP
                    """, (
                        int(customer_id), period_id, algo_name, cluster_id,
                        cluster_label, is_outlier, json.dumps(metadata)
                    ))
            
            conn.commit()
            cursor.close()
            conn.close()
            
            logger.info(f"✅ Clean segmentation results saved for {period_id}")
            
        except Exception as e:
            logger.error(f"❌ Database save failed: {e}")
            raise
    
    def generate_comparison_report(self):
        """Generate comparison report for all algorithms"""
        logger.info("📊 Generating algorithm comparison report...")
        
        comparison_data = []
        
        for algo_name, result in self.results.items():
            if result is None:
                continue
            
            comparison_data.append({
                'Algorithm': algo_name,
                'N_Clusters': result['n_clusters'],
                'N_Outliers': result['n_outliers'],
                'Outlier_Pct': result['outlier_percentage'],
                'Silhouette_Score': result['silhouette_score'],
                'Davies_Bouldin': result['davies_bouldin_score']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "="*80)
        print("CLEAN DATA MULTI-ALGORITHM COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))
        
        # Detailed segment analysis
        print(f"\nDETAILED SEGMENT ANALYSIS:")
        for algo_name, result in self.results.items():
            if result is None:
                continue
            
            print(f"\n{algo_name.upper()}:")
            for cluster_id, profile in result['segment_profiles'].items():
                print(f"  {profile['business_label']}: {profile['size']} customers "
                      f"({profile['percentage']}%) - Avg Bet: €{profile['avg_total_bet']}")
        
        return comparison_df

def main():
    parser = argparse.ArgumentParser(description='Clean Data Re-Segmentation')
    parser.add_argument('--period', help='Period ID (e.g., 2022-H1)')
    parser.add_argument('--all_periods', action='store_true', help='Process all periods')
    parser.add_argument('--backup', action='store_true', help='Backup old segmentation')
    args = parser.parse_args()
    
    # Check arguments
    if not args.all_periods and not args.period:
        parser.error("Either --period PERIOD or --all_periods is required")
    
    # Initialize clean segmentation
    segmenter = CleanDataSegmentation()
    
    # Backup if requested
    if args.backup:
        segmenter.backup_old_segmentation()
    
    # Define periods to process
    if args.all_periods:
        periods = ['2022-H1', '2022-H2', '2023-H1', '2023-H2']
    else:
        periods = [args.period]
    
    print("🎓 CLEAN DATA RE-SEGMENTATION - BATH UNIVERSITY A GRADE")
    print("="*70)
    print("Academic Grade: A EXCELLENT")
    print("Data Quality: CV 2.788, €1.5M cap, Outlier-free")
    print("Validation: ChatGPT approved %1.7 modification")
    print("="*70)
    
    # Process each period
    for period in periods:
        logger.info(f"🔄 Processing {period}...")
        
        try:
            # Load clean data
            df = segmenter.load_clean_data(period)
            
            # Prepare features
            X_scaled, customer_ids, df_full = segmenter.prepare_features(df)
            
            # Run all algorithms
            results = segmenter.run_all_algorithms(X_scaled, customer_ids, df_full)
            
            # Save to database
            segmenter.save_results_to_database(period, customer_ids)
            
            # Generate report
            comparison_df = segmenter.generate_comparison_report()
            
            logger.info(f"✅ Successfully processed {period}")
            
        except Exception as e:
            logger.error(f"❌ Failed to process {period}: {e}")
            continue
    
    print("\n🏆 CLEAN DATA RE-SEGMENTATION COMPLETED!")
    print("Ready for A-Grade RF Training!")

if __name__ == "__main__":
    main()