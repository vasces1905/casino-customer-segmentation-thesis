# multi_algorithm_segmentation.py - Compare Multiple Clustering Algorithms
import numpy as np
import pandas as pd
import psycopg2
import json
import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from sklearn.manifold import TSNE
import joblib
from datetime import datetime
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultiAlgorithmSegmentation:
    def __init__(self):
        self.scaler = StandardScaler()
        self.algorithms = {
            'kmeans': KMeans(n_clusters=4, random_state=42, n_init=10),
            'dbscan': DBSCAN(eps=0.5, min_samples=5),
            'gaussian_mixture': GaussianMixture(n_components=4, random_state=42),
            'hierarchical': AgglomerativeClustering(n_clusters=4)
        }
        self.results = {}
        self.feature_columns = None
        
    def load_data(self, period_id: str) -> pd.DataFrame:
        """Load customer features data for the specified period"""
        conn = psycopg2.connect(
            host="localhost",
            database="casino_research", 
            user="researcher",
            password="academic_password_2024"
        )
        
        query = f"""
        SELECT customer_id, total_bet, avg_bet, loss_rate, total_sessions,
               days_since_last_visit, session_duration_volatility,
               loss_chasing_score, sessions_last_30d, bet_trend_ratio
        FROM casino_data.customer_features 
        WHERE analysis_period = '{period_id}'
        AND total_bet > 0
        ORDER BY customer_id
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"Loaded {len(df)} customers for {period_id}")
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare and normalize features for clustering"""
        feature_columns = [
            'total_bet', 'avg_bet', 'loss_rate', 'total_sessions',
            'days_since_last_visit', 'session_duration_volatility', 
            'loss_chasing_score', 'sessions_last_30d', 'bet_trend_ratio'
        ]
        
        # Select features and handle missing values
        X = df[feature_columns].fillna(df[feature_columns].median())
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        self.feature_columns = feature_columns
        logger.info(f"Prepared {len(feature_columns)} features for clustering")
        
        return X_scaled, df['customer_id'].values
    
    def run_clustering_algorithms(self, X_scaled: np.ndarray, customer_ids: np.ndarray):
        """Run all clustering algorithms and collect results"""
        results = {}
        
        for name, algorithm in self.algorithms.items():
            logger.info(f"Running {name} clustering...")
            
            try:
                # Fit the algorithm
                if name == 'gaussian_mixture':
                    labels = algorithm.fit_predict(X_scaled)
                else:
                    labels = algorithm.fit_predict(X_scaled)
                
                # Calculate metrics
                n_clusters = len(np.unique(labels[labels != -1]))  # Exclude outliers for DBSCAN
                n_outliers = np.sum(labels == -1)
                
                # Silhouette score (only if we have more than 1 cluster)
                if n_clusters > 1 and n_outliers < len(labels) - 1:
                    # For silhouette score, exclude outliers
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
                
                # Store results
                results[name] = {
                    'labels': labels,
                    'n_clusters': n_clusters,
                    'n_outliers': n_outliers,
                    'silhouette_score': silhouette,
                    'davies_bouldin_score': davies_bouldin,
                    'algorithm': algorithm
                }
                
                logger.info(f"{name}: {n_clusters} clusters, {n_outliers} outliers, "
                           f"silhouette: {silhouette:.4f}")
                
            except Exception as e:
                logger.error(f"Error running {name}: {e}")
                results[name] = None
        
        return results
    
    def analyze_segment_characteristics(self, df: pd.DataFrame, X_scaled: np.ndarray, 
                                      results: dict) -> dict:
        """Analyze characteristics of segments for each algorithm"""
        analysis = {}
        
        for algo_name, result in results.items():
            if result is None:
                continue
                
            labels = result['labels']
            
            # Add labels to dataframe copy
            df_temp = df.copy()
            df_temp['cluster'] = labels
            
            # Analyze each cluster
            cluster_analysis = {}
            unique_labels = np.unique(labels)
            
            for cluster_id in unique_labels:
                cluster_mask = labels == cluster_id
                cluster_data = df_temp[cluster_mask]
                
                if len(cluster_data) == 0:
                    continue
                
                cluster_analysis[int(cluster_id)] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(df) * 100,
                    'avg_total_bet': float(cluster_data['total_bet'].mean()),
                    'avg_loss_rate': float(cluster_data['loss_rate'].mean()),
                    'avg_loss_chasing': float(cluster_data['loss_chasing_score'].mean()),
                    'avg_sessions': float(cluster_data['total_sessions'].mean()),
                }
            
            analysis[algo_name] = {
                'clusters': cluster_analysis,
                'metrics': {
                    'silhouette_score': result['silhouette_score'],
                    'davies_bouldin_score': result['davies_bouldin_score'],
                    'n_clusters': result['n_clusters'],
                    'n_outliers': result['n_outliers']
                }
            }
        
        return analysis
    
    def compare_algorithms(self, results: dict) -> pd.DataFrame:
        """Create comparison summary of all algorithms"""
        comparison_data = []
        
        for algo_name, result in results.items():
            if result is None:
                continue
                
            comparison_data.append({
                'Algorithm': algo_name,
                'N_Clusters': result['n_clusters'],
                'N_Outliers': result['n_outliers'],
                'Silhouette_Score': round(result['silhouette_score'], 4) if result['silhouette_score'] != -1 else 'N/A',
                'Davies_Bouldin_Score': round(result['davies_bouldin_score'], 4) if result['davies_bouldin_score'] != -1 else 'N/A',
                'Outlier_Percentage': round(result['n_outliers'] / (len(result['labels'])) * 100, 2)
            })
        
        return pd.DataFrame(comparison_data)
    
    def save_results_to_database(self, period_id: str, customer_ids: np.ndarray, 
                                results: dict, analysis: dict):
        """Save multi-algorithm results to database"""
        conn = psycopg2.connect(
            host="localhost",
            database="casino_research",
            user="researcher", 
            password="academic_password_2024"
        )
        cur = conn.cursor()
        
        # Create table if not exists
        cur.execute("""
            CREATE TABLE IF NOT EXISTS casino_data.multi_algorithm_segments (
                customer_id BIGINT,
                period_id TEXT,
                algorithm_name TEXT,
                cluster_id INTEGER,
                cluster_label TEXT,
                is_outlier BOOLEAN,
                algorithm_metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (customer_id, period_id, algorithm_name)
            )
        """)
        
        # Insert results for each algorithm
        for algo_name, result in results.items():
            if result is None:
                continue
                
            labels = result['labels']
            
            for i, customer_id in enumerate(customer_ids):
                cluster_id = int(labels[i])
                is_outlier = cluster_id == -1
                cluster_label = f"{algo_name}_cluster_{cluster_id}" if not is_outlier else f"{algo_name}_outlier"
                
                metadata = {
                    'silhouette_score': result['silhouette_score'],
                    'davies_bouldin_score': result['davies_bouldin_score'],
                    'n_clusters': result['n_clusters'],
                    'algorithm_type': algo_name
                }
                
                cur.execute("""
                    INSERT INTO casino_data.multi_algorithm_segments
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
        cur.close()
        conn.close()
        
        logger.info(f"Saved multi-algorithm results for {period_id}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', required=True, help='Period ID (e.g., 2022-H1)')
    parser.add_argument('--output_dir', default='models/multi_algorithm', help='Output directory')
    args = parser.parse_args()
    
    # Initialize multi-algorithm segmentation
    mas = MultiAlgorithmSegmentation()
    
    # Load and prepare data
    logger.info(f"Starting multi-algorithm analysis for {args.period}")
    df = mas.load_data(args.period)
    X_scaled, customer_ids = mas.prepare_features(df)
    
    # Run all clustering algorithms
    results = mas.run_clustering_algorithms(X_scaled, customer_ids)
    
    # Analyze segment characteristics
    analysis = mas.analyze_segment_characteristics(df, X_scaled, results)
    
    # Create comparison summary
    comparison_df = mas.compare_algorithms(results)
    
    # Print results
    print("\n" + "="*80)
    print(f"MULTI-ALGORITHM CLUSTERING COMPARISON - {args.period}")
    print("="*80)
    print(comparison_df.to_string(index=False))
    
    print(f"\nDetailed Analysis:")
    for algo_name, algo_analysis in analysis.items():
        print(f"\n{algo_name.upper()}:")
        print(f"  Metrics: {algo_analysis['metrics']}")
        print(f"  Clusters: {len(algo_analysis['clusters'])}")
        for cluster_id, cluster_info in algo_analysis['clusters'].items():
            print(f"    Cluster {cluster_id}: {cluster_info['size']} customers "
                  f"({cluster_info['percentage']:.1f}%) - "
                  f"Avg Bet: €{cluster_info['avg_total_bet']:.2f}")
    
    # Save results to database
    mas.save_results_to_database(args.period, customer_ids, results, analysis)
    
    # Save models
    import os
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump({
        'results': results,
        'analysis': analysis,
        'comparison': comparison_df,
        'scaler': mas.scaler,
        'feature_columns': mas.feature_columns
    }, f"{args.output_dir}/multi_algorithm_{args.period}.pkl")
    
    logger.info(f"✅ Multi-algorithm analysis completed for {args.period}")

if __name__ == "__main__":
    main()