# segmentation.py - FIXED VERSION - All Bugs Resolved - It's worked
import numpy as np
import pandas as pd
import psycopg2
import argparse
import json
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, Any
import logging
import joblib
from datetime import datetime

def clean_nan(obj):
    """NaN, inf gibi PostgreSQL'in JSON formatını bozan değerleri temizler."""
    if isinstance(obj, dict):
        return {k: clean_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan(i) for i in obj]
    elif isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
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
        
        # FIXED: Clean metadata creation
        self.model_metadata.update({
            'fit_date': datetime.now().isoformat(),  # FIXED: Use isoformat() instead of str()
            'n_samples': int(len(df)),
            'n_features': int(len(selected_features.columns)),
            'feature_list': list(selected_features.columns),
            'silhouette_score': float(silhouette),
            'davies_bouldin_score': float(davies)
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
            if group.empty:
                continue

            # FIXED: Safe avg_session_value calculation
            try:
                total_sessions_safe = group['total_sessions'].replace(0, 1)
                session_values = group['total_bet'] / total_sessions_safe
                avg_session_val = float(session_values.mean())
                if np.isnan(avg_session_val) or np.isinf(avg_session_val):
                    avg_session_val = 0.0
            except Exception as e:
                logger.warning(f"Error calculating avg_session_value for segment {s}: {e}")
                avg_session_val = 0.0

            profile = {
                'segment_id': int(s),
                'size': int(len(group)),
                'percentage': float(len(group) / len(df) * 100),
                'avg_total_wagered': float(group['total_bet'].mean()),
                'avg_loss_rate': float(group['loss_rate'].mean()),
                'avg_session_value': avg_session_val,
                'avg_sessions': float(group['total_sessions'].mean()),
                'avg_days_since_visit': float(group['days_since_last_visit'].mean()),
                'avg_loss_chasing_score': float(group['loss_chasing_score'].mean()),
                'avg_volatility': float(group['session_duration_volatility'].mean()),
                'high_risk_percentage': float((group['loss_chasing_score'] > 0.3).mean() * 100),
            }

            profile['business_label'] = self._assign_segment_label(profile)
            profiles[s] = profile

        self.segment_profiles = profiles

    def _assign_segment_label(self, profile: Dict) -> str:
        # FIXED: More balanced thresholds for better segment distribution
        if profile['avg_total_wagered'] > 5000 and profile['avg_sessions'] > 10:
            return "High_Value_Player"
        elif profile['avg_loss_chasing_score'] > 70 or profile['high_risk_percentage'] > 60:
            return "At_Risk_Player"  # FIXED: Higher threshold (30->60)
        elif profile['avg_sessions'] > 3 and profile['avg_total_wagered'] > 800:
            return "Regular_Player"  # FIXED: Lower threshold for more inclusion
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

def save_segment_metadata_to_database(results, period_id):
    conn = get_database_connection()
    cur = conn.cursor()
    
    try:
        for row in results['segment_profiles']:
            cluster_label = row['business_label']
            # FIXED: Clean metadata before JSON serialization
            clean_metadata = clean_nan(results['metadata'])
            model_metadata_json = json.dumps(clean_metadata, ensure_ascii=False)

            cur.execute("""
                INSERT INTO casino_data.kmeans_segment_metadata (
                    period_id, cluster_label, model_metadata
                )
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (period_id, cluster_label) DO UPDATE
                SET model_metadata = EXCLUDED.model_metadata,
                    created_at = CURRENT_TIMESTAMP
            """, (period_id, cluster_label, model_metadata_json))

        conn.commit()
        logger.info("Segment-level metadata saved to DB.")
    except Exception as e:
        logger.error(f"Error saving segment metadata: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

def save_customer_segments_to_database(results, period_id):
    conn = get_database_connection()
    cur = conn.cursor()
    
    try:
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

            # FIXED: Match exact table structure positions
            cur.execute("""
            INSERT INTO casino_data.kmeans_segments (
                customer_id, period_id, cluster_id, cluster_label,
                silhouette_score, distance_to_centroid, model_metadata,
                kmeans_version, segment_data, avg_session_from_metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s::jsonb, %s)
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
                cluster_id,
                cluster_label,
                silhouette_score,
                distance,
                metadata_json,
                1,  # kmeans_version - FIXED: moved to correct position
                segment_data_json,
                avg_session_val  # FIXED: moved to end
            ))

        conn.commit()
        logger.info(f"Saved {len(results['customer_segments'])} customer segments to DB.")
    except Exception as e:
        logger.error(f"Error saving customer segments: {e}")
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

# FIXED: Better NumPy converter
def convert_numpy(o):
    if isinstance(o, (np.integer, np.int64, np.int32)):
        return int(o)
    elif isinstance(o, (np.floating, np.float64, np.float32)):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    elif isinstance(o, np.bool_):
        return bool(o)
    elif pd.isna(o):
        return None
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--period', required=True)
    parser.add_argument('--n_clusters', type=int, default=4)
    parser.add_argument('--output_dir', default='models/segmentation')
    args = parser.parse_args()

    # 1. Load & Train
    logger.info(f"Processing period: {args.period}")
    df = load_period_data(args.period)
    
    # FIXED: Check if data is loaded correctly
    if df.empty:
        logger.error(f"No data loaded for period {args.period}")
        return
    
    logger.info(f"DataFrame columns: {df.columns.tolist()}")
    logger.info(f"DataFrame shape: {df.shape}")

    model = CustomerSegmentation(n_clusters=args.n_clusters)
    model.fit(df)

    labels = model.predict(df)
    segment_profiles = model.get_segment_summary()
    segment_summary_dict = segment_profiles.to_dict(orient="records")

    # 2. FIXED: Safer avg_session_value calculation
    try:
        total_bet_sum = float(df["total_bet"].sum())
        total_sessions_sum = float(df["total_sessions"].sum())
        avg_session_value = total_bet_sum / max(total_sessions_sum, 1)
        model.model_metadata["avg_session_value"] = avg_session_value
        logger.info(f"Calculated avg_session_value: {avg_session_value}")
    except Exception as e:
        logger.error(f"Error calculating avg_session_value: {e}")
        model.model_metadata["avg_session_value"] = 0.0

    # 3. Save model
    model_path = f"{args.output_dir}/segmentation_model_{args.period}.pkl"
    model.save_model(model_path)

    # 4. FIXED: Better customer segment preparation
    customer_segments = []
    for idx, row in df.iterrows():
        try:
            segment_id = int(labels[idx])
            
            # FIXED: Ensure customer_id is properly handled
            if 'customer_id' not in df.columns:
                logger.error("customer_id column not found in DataFrame!")
                continue
                
            customer_id_raw = row['customer_id']
            if pd.isna(customer_id_raw):
                logger.warning(f"Skipping row {idx} with NULL customer_id")
                continue
            
            customer_id = int(customer_id_raw)
            
            # Get clean segment profile
            cluster_row = clean_nan(model.segment_profiles[segment_id])
            cluster_label = cluster_row['business_label']
            
            # FIXED: Clean segment data JSON
            segment_data_json = json.dumps(cluster_row, default=convert_numpy, ensure_ascii=False)

            # FIXED: Safe personal avg_session_value calculation
            try:
                personal_total_bet = float(row['total_bet']) if not pd.isna(row['total_bet']) else 0.0
                personal_total_sessions = float(row['total_sessions']) if not pd.isna(row['total_sessions']) else 1.0
                personal_avg_session_value = personal_total_bet / max(personal_total_sessions, 1)
            except Exception:
                personal_avg_session_value = 0.0

            # FIXED: Create clean, minimal metadata
            clean_metadata = {
                "fit_date": model.model_metadata["fit_date"],
                "silhouette_score": float(model.model_metadata["silhouette_score"]),
                "n_samples": int(model.model_metadata["n_samples"]),
                "model_type": model.model_metadata["model_type"],
                "created_by": model.model_metadata["created_by"]
            }
            
            metadata_json = json.dumps(clean_metadata, default=convert_numpy, ensure_ascii=False)

            customer_segments.append((
                customer_id,
                args.period,
                segment_id,
                cluster_label,
                float(model.model_metadata['silhouette_score']),
                0.0,  # distance_to_centroid
                metadata_json,
                personal_avg_session_value,
                segment_data_json
            ))

        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            continue

    # 5. FIXED: Filter and validate results
    customer_segments = [x for x in customer_segments if x[0] is not None]
    logger.info(f"Prepared {len(customer_segments)} valid customer segments")

    if not customer_segments:
        logger.error("No valid customer segments prepared!")
        return

    # FIXED: Clean results structure
    results = {
        'customer_segments': customer_segments,
        'segment_profiles': segment_summary_dict,
        'metadata': clean_nan(model.model_metadata)
    }

    # 6. Enhanced debugging
    logger.info("DEBUG - Model metadata:")
    print(json.dumps(results['metadata'], indent=2, default=convert_numpy))
    
    logger.info("DEBUG - First customer segment:")
    print(customer_segments[0])

    # 7. Save to database
    try:
        save_customer_segments_to_database(results, args.period)
        save_segment_metadata_to_database(results, args.period)
        logger.info(f"✅ Successfully processed period {args.period}")
    except Exception as e:
        logger.error(f"❌ Database save failed: {e}")
        raise

if __name__ == "__main__":
    main()