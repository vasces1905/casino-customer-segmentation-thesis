# segmentation.py - Final Cleaned Version for Periodic KMeans Clustering

import numpy as np
import pandas as pd
import psycopg2
import argparse
import json
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict
import logging
import joblib
from datetime import datetime

# Logger setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class CustomerSegmentation:
    def __init__(self, n_clusters: int = 4, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.segment_profiles = None
        self.model_metadata = {
            "model_type": "KMeans_Clustering",
            "created_by": "Muhammed Yavuzhan CANLI",
            "academic_purpose": "Customer behavioral segmentation",
            "ethics_ref": "10351-12382",
            "institution": "University of Bath"
        }

    def select_features(self, df: pd.DataFrame, feature_list=None):
        if feature_list is None:
            feature_list = [
                'total_bet', 'avg_bet', 'loss_rate', 'total_sessions',
                'days_since_last_visit', 'session_duration_volatility',
                'loss_chasing_score', 'sessions_last_30d', 'bet_trend_ratio'
            ]
        available_features = [f for f in feature_list if f in df.columns]
        self.feature_columns = available_features
        logger.info(f"Selected features: {available_features}")
        return df[available_features]

    def fit(self, df, selected_features: pd.DataFrame = None, feature_list=None):
        logger.info(f"Starting KMeans clustering for {len(df)} customers")
        if selected_features is None:
            selected_features = self.select_features(df, feature_list)

        X_clean = selected_features.fillna(selected_features.median())
        X_scaled = self.scaler.fit_transform(X_clean)
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10, max_iter=300)
        labels = self.kmeans.fit_predict(X_scaled)

        silhouette = float(silhouette_score(X_scaled, labels))
        davies = float(davies_bouldin_score(X_scaled, labels))

        logger.info(f"Silhouette Score: {silhouette:.4f} | Davies-Bouldin: {davies:.4f}")

        self._create_segment_profiles(df, selected_features, labels)
        self.model_metadata.update({
            'fit_date': str(datetime.now()),
            'model_type': 'KMeans_Clustering',
            'created_by': 'Muhammed Yavuzhan CANLI',
            'ethics_ref': '10351-12382',
            'institution': 'University of Bath',
            'academic_purpose': 'Customer behavioral segmentation',
            'n_samples': len(df),
            'n_features': len(selected_features.columns),
            'feature_list': list(selected_features.columns),
            'silhouette_score': silhouette,
            'davies_bouldin_score': davies,
            'avg_session_value': float(segment_data.get('avg_session_value', 
                segment_data['total_wagered']/segment_data['total_sessions']).mean()),
               # It receives a total of bet / total session for all customers, then calculates its average.
        })
        return self


    def predict(self, df):
        X = df[self.feature_columns].fillna(df[self.feature_columns].median())
        return self.kmeans.predict(self.scaler.transform(X))

    def _create_segment_profiles(self, df: pd.DataFrame, X: pd.DataFrame, labels: np.ndarray):
        df_seg = df.copy()
        df_seg['segment'] = labels
        profiles = {}
        for s in range(self.n_clusters):
            group = df_seg[df_seg['segment'] == s]
            profiles[s] = {
                'segment_id': s,
                'size': len(group),
                'percentage': len(group) / len(df) * 100,
                'avg_total_wagered': float(group['total_bet'].mean()),
                'avg_loss_rate': float(group['loss_rate'].mean()),
                #'avg_session_value': float((group['total_bet'] / group['total_sessions']).mean()),
                'avg_session_value': float((group['total_bet'] / group['total_sessions']).mean()),
                'avg_sessions': float(group['total_sessions'].mean()),
                'avg_days_since_visit': float(group['days_since_last_visit'].mean()),
                'avg_loss_chasing_score': float(group['loss_chasing_score'].mean()),
                'avg_volatility': float(group['session_duration_volatility'].mean()),
                'high_risk_percentage': float((group['loss_chasing_score'] > 0.3).mean() * 100),
            }

            profiles[s]['business_label'] = self._assign_segment_label(profiles[s])
        self.segment_profiles = profiles

    def _assign_segment_label(self, profile: Dict) -> str:
        if profile['avg_total_wagered'] > 3000 and profile['avg_sessions'] > 8:
            return "High_Value_Player"
        elif profile['avg_loss_chasing_score'] > 50 or profile['high_risk_percentage'] > 30:
            return "At_Risk_Player"
        elif profile['avg_sessions'] > 5 and profile['avg_total_wagered'] > 1000:
            return "Regular_Player"
        else:
            return "Casual_Player"

    def get_segment_summary(self):
        return pd.DataFrame.from_dict(self.segment_profiles, orient='index')

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'kmeans': self.kmeans,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'segment_profiles': self.segment_profiles,
            'metadata': self.model_metadata
        }, path)
        logger.info(f"Model saved to {path}")


def get_database_connection():
    return psycopg2.connect(
        host="localhost",
        database="casino_research",
        user="researcher",
        password="academic_password_2024"
    )

def load_period_data(period_id: str) -> pd.DataFrame:
    view = f"casino_data.kmeans_export_{period_id.lower().replace('-', '_')}"
    conn = get_database_connection()
    try:
        df = pd.read_sql_query(f"SELECT * FROM {view}", conn)
        logger.info(f"Loaded {len(df)} records from {view}")
        return df
    finally:
        conn.close()
        
# - updated 20 July 2025
def save_results_to_database(results: Dict, period_id: str):
    conn = get_database_connection()
    try:
        cursor = conn.cursor()
        for customer_data in results['customer_segments']:
            cursor.execute("""
                INSERT INTO casino_data.kmeans_segments 
                (customer_id, period_id, cluster_id, cluster_label, 
                 silhouette_score, distance_to_centroid, model_metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (customer_id, period_id, kmeans_version) 
                DO UPDATE SET
                    cluster_id = EXCLUDED.cluster_id,
                    cluster_label = EXCLUDED.cluster_label,
                    silhouette_score = EXCLUDED.silhouette_score,
                    distance_to_centroid = EXCLUDED.distance_to_centroid,
                    model_metadata = EXCLUDED.model_metadata,
                    created_at = CURRENT_TIMESTAMP
            """, customer_data)
        conn.commit()
        logger.info(f"Saved {len(results['customer_segments'])} customer segments to DB for period {period_id}")
    except Exception as e:
        conn.rollback()
        logger.error(f"DB insert failed: {e}")
        raise
    finally:
        cursor.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', required=True)
    parser.add_argument('--n_clusters', type=int, default=4)
    parser.add_argument('--output_dir', default='models/segmentation')
    args = parser.parse_args()

    df = load_period_data(args.period)
    model = CustomerSegmentation(n_clusters=args.n_clusters)
    model.fit(df)

    labels = model.predict(df)
    summary = model.get_segment_summary()
    logger.info("Segment Summary:\n" + summary.to_string())

    model_path = f"{args.output_dir}/segmentation_model_{args.period}.pkl"
    model.save_model(model_path)
    logger.info(f"Finished segmentation for {args.period}")
    
        # Prepare DB output - updated 20 July 2025
    labels = model.predict(df)
    customer_segments = []
    for idx, row in df.iterrows():
        
        customer_segments.append((
            int(row['customer_id']),
            args.period,
            int(labels[idx]),
            model.segment_profiles[labels[idx]]['business_label'],
            float(model.model_metadata['silhouette_score']),
            float(0.0),
            json.dumps(model.model_metadata)
        ))

    results = {
        'customer_segments': customer_segments,
        'segment_profiles': model.segment_profiles,
        'metadata': model.model_metadata
    }

    save_results_to_database(results, args.period)

if __name__ == "__main__":
    main()
