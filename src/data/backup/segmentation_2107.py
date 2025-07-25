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

def clean_nan(obj):
    """NaN, inf gibi PostgreSQL’in JSON formatını bozan değerleri temizler."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(i) for i in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


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

    # FIXED: Remove problematic lines from model_metadata
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
    
    # FIXED: Remove problematic avg_session_value calculation from metadata
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
            'davies_bouldin_score': davies
            # REMOVED: avg_session_value calculation (moved to segment profiles)
        })
        return self


    def predict(self, df):
        X = df[self.feature_columns].fillna(df[self.feature_columns].median())
        return self.kmeans.predict(self.scaler.transform(X))

# FIXED: _create_segment_profiles with proper avg_session_value calculation
    def _create_segment_profiles(self, df: pd.DataFrame, X: pd.DataFrame, labels: np.ndarray):
        df_seg = df.copy()
        df_seg['segment'] = labels
        profiles = {}

        for s in range(self.n_clusters):
            group = df_seg[df_seg['segment'] == s]
            if group.empty:
                continue

            try:
                session_values = group['total_bet'] / group['total_sessions'].replace(0, 1)
                avg_session_val = float(session_values.mean())
            except Exception:
                avg_session_val = float('nan')

            profile = {
                'segment_id': s,
                'size': len(group),
                'percentage': len(group) / len(df) * 100,
                'avg_total_wagered': float(group['total_bet'].mean()),
                'avg_loss_rate': float(group['loss_rate'].mean()),
                'avg_session_value': avg_session_val,
                'avg_sessions': float(group['total_sessions'].mean()),
                'avg_days_since_visit': float(group['days_since_last_visit'].mean()),
                'avg_loss_chasing_score': float(group['loss_chasing_score'].mean()),
                'avg_volatility': float(group['session_duration_volatility'].mean()),
                'high_risk_percentage': float((group['loss_chasing_score'] > 0.3).mean() * 100),
            }

            # complete business_label appointment here:
            profile['business_label'] = self._assign_segment_label(profile)

            profiles[s] = profile

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
# deleted !! def save_results_to_database(results, period_id):
def save_segment_metadata_to_database(results, period_id):
    conn = psycopg2.connect(
        dbname="casino_research",
        user="researcher",
        password="academic_password_2024",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    for row in results['segment_profiles']:
        cluster_label = row['business_label']
        model_metadata_json = json.dumps(results['metadata'], default=convert_numpy)

        cur.execute("""
            INSERT INTO casino_data.kmeans_segment_metadata (
                period_id, cluster_label, model_metadata
            )
            VALUES (%s, %s, %s)
            ON CONFLICT (period_id, cluster_label) DO UPDATE
            SET model_metadata = EXCLUDED.model_metadata,
                created_at = CURRENT_TIMESTAMP
        """, (period_id, cluster_label, model_metadata_json))

    conn.commit()
    cur.close()
    conn.close()
    logger.info("Segment-level metadata saved to DB.")


# It is customer -based version of the function (ie fills the customer_id field).
def save_customer_segments_to_database(results, period_id):
    conn = psycopg2.connect(
        dbname="casino_research",
        user="researcher",
        password="academic_password_2024",
        host="localhost",
        port="5432"
    )
    cur = conn.cursor()

    for segment in results['customer_segments']:
        (
            customer_id,
            period_id_val,
            cluster_id,
            cluster_label,
            silhouette_score,
            distance,
            metadata_json,
            avg_session_val,
            segment_data_json
        ) = segment

        cur.execute("""
        INSERT INTO casino_data.kmeans_segments (
            customer_id, period_id, kmeans_version, cluster_id, cluster_label,
            silhouette_score, distance_to_centroid,
            model_metadata, avg_session_from_metadata, segment_data
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (customer_id, period_id, kmeans_version) DO UPDATE
        SET cluster_id = EXCLUDED.cluster_id,
            cluster_label = EXCLUDED.cluster_label,
            silhouette_score = EXCLUDED.silhouette_score,
            distance_to_centroid = EXCLUDED.distance_to_centroid,
            model_metadata = EXCLUDED.model_metadata,
            segment_data = EXCLUDED.segment_data,
            avg_session_from_metadata = EXCLUDED.avg_session_from_metadata,
            created_at = CURRENT_TIMESTAMP
        """, (
            customer_id,
            period_id_val,
            1,  # ← kmeans_version artık int olmalı, "v1" hataydı
            cluster_id,
            cluster_label,
            silhouette_score,
            distance,
            metadata_json,
            avg_session_val,
            segment_data_json,
            
        ))

    conn.commit()
    cur.close()
    conn.close()
    logger.info(f"Saved {len(results['customer_segments'])} customer segments to DB.")


    # NumPy compatible serialize - 20 july 2025
def convert_numpy(o):
    if isinstance(o, (np.integer, np.floating)):
        return o.item()
    elif isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', required=True)
    parser.add_argument('--n_clusters', type=int, default=4)
    parser.add_argument('--output_dir', default='models/segmentation')
    args = parser.parse_args()

    # 1. Load & Train
    df = load_period_data(args.period)
    model = CustomerSegmentation(n_clusters=args.n_clusters)
    model.fit(df)

    labels = model.predict(df)
    segment_profiles = model.get_segment_summary()
    segment_summary_dict = segment_profiles.to_dict(orient="records")

    # 2. Add only high-level info to metadata (NO segment_summary here)
    model.model_metadata["avg_session_value"] = float(df["total_bet"].sum()) / max(float(df["total_sessions"].sum()), 1)

    # 3. Save model
    model_path = f"{args.output_dir}/segmentation_model_{args.period}.pkl"
    model.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

    # 4. Prepare customer-specific segments
    customer_segments = []
    for idx, row in df.iterrows():
        segment_id = int(labels[idx])
        cluster_row = clean_nan(model.segment_profiles[segment_id])  # only this customer's segment
        cluster_label = cluster_row['business_label']
        segment_data = json.dumps(cluster_row, default=convert_numpy)

        try:
            personal_avg_session_value = float(row['total_bet']) / max(float(row['total_sessions']), 1)
        except Exception:
            personal_avg_session_value = None

        customer_segments.append((
            int(row['customer_id']),
            args.period,
            segment_id,
            cluster_label,
            float(model.model_metadata['silhouette_score']),
            0.0,  # distance_to_centroid (optional for now)
            json.dumps({  # cleaned, shallow metadata
                "fit_date": model.model_metadata["fit_date"],
                "silhouette_score": model.model_metadata["silhouette_score"],
                "n_samples": model.model_metadata["n_samples"]
            }, default=convert_numpy),
            personal_avg_session_value,
            segment_data
        ))

    # 5. Finalize
    customer_segments = [x for x in customer_segments if x[0] is not None]

    results = {
        'customer_segments': customer_segments,
        'segment_profiles': segment_summary_dict,
        'metadata': clean_nan(model.model_metadata)
    }

    save_customer_segments_to_database(results, args.period)   # Individual records
    save_segment_metadata_to_database(results, args.period)    # Only one per cluster

    logger.info("DEBUG - Sample customer_segments[0]:")
    print(customer_segments[0])
    print(df.columns)
    print(df[['customer_id']].head())
    print(type(row['customer_id']), row['customer_id'])
    
    print("First few customer_segments entries:")
    for x in customer_segments[:5]:
        print(x)

if __name__ == "__main__":
    main()
