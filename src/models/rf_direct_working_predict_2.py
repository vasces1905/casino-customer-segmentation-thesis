"""
Random Forest Direct Prediction System
University of Bath - MSc Computer Science Dissertation
Student: Muhammed Yavuzhan CANLI | Ethics Ref: 10351-12382

This module implements direct prediction using pre-trained Random Forest models
with feature engineering and validation guards for academic research purposes.
"""

import pandas as pd
import numpy as np
import joblib
import psycopg2
from datetime import datetime
import warnings
import argparse, sys, os, json, time
import argparse, time, numpy as np
from pathlib import Path

warnings.filterwarnings('ignore')

# --- PATCH: load_working_model(period) ---
def load_working_model(period: str):
    """
    Load Random Forest model package for specified analysis period.
    Falls back to 2023-H2 model if requested period is not available.
    
    Args:
        period (str): Analysis period identifier (e.g., '2023-H1', '2023-H2')
    
    Returns:
        tuple: (rf_model, scaler, label_encoder, feature_names)
    """
    here = Path(__file__).resolve().parent  # .../src/models
    def cand(p):
        return [
            here / "models" / "optimized_rf" / f"optimized_kmeans_dbscan_rf_{p}.pkl",
            here / "optimized_rf" / f"optimized_kmeans_dbscan_rf_{p}.pkl",
            here.parent / "models" / "optimized_rf" / f"optimized_kmeans_dbscan_rf_{p}.pkl",
        ]

    candidates = cand(period)
    fallback = "2023-H2"
    if period != fallback:
        candidates += cand(fallback)

    for path in candidates:
        if path.exists():
            print(f"Loading model from: {path}")
            pkg = joblib.load(path)
            rf = pkg["rf_model"]
            scaler = pkg["scaler"]
            label_encoder = pkg["label_encoder"]
            feature_names = pkg.get("feature_names") or pkg.get("expected_features")
            if feature_names is None:
                raise KeyError("feature_names/expected_features missing in model package")
            return rf, scaler, label_encoder, feature_names

    tried = "\n".join(str(x) for x in candidates)
    raise FileNotFoundError(f"RF model pkl not found for period={period}.\nTried:\n{tried}")


# --- PATCH: load_customer_data(period) ---
import psycopg2, pandas as pd

def load_customer_data(period: str):
    """
    Load customer data from engineered features view for specified period.
    This function accesses the pre-computed engineered features database view
    that contains all necessary customer behavioral metrics.
    
    Args:
        period (str): Analysis period identifier
    
    Returns:
        pandas.DataFrame: Customer data with engineered features
    """
    conn = psycopg2.connect(host="localhost", dbname="casino_research",
                            user="researcher", password="academic_password_2024", port=5432)
    q = "SELECT * FROM casino_data.engineered_features_all WHERE period = %s"
    df = pd.read_sql(q, conn, params=[period])
    conn.close()
    return df

# --- PATCH: engineered features + guard ---

def create_engineered_features(df):
    """
    Create engineered features from base customer data columns.
    Only generates derived features if source columns are present in dataset.
    This ensures safe feature engineering without data corruption.
    
    Args:
        df (pandas.DataFrame): Input customer data
    
    Returns:
        pandas.DataFrame: Data with additional engineered features
    """
    if "avg_bet" in df and "log_avg_bet" not in df:
        df["log_avg_bet"] = np.log1p(df["avg_bet"].clip(lower=0))
    if "total_bet" in df and "log_total_bet" not in df:
        df["log_total_bet"] = np.log1p(df["total_bet"].clip(lower=0))
    if "total_sessions" in df and "log_sessions" not in df:
        df["log_sessions"] = np.log1p(df["total_sessions"].clip(lower=0))
    if "total_bet" in df and "is_millionaire" not in df:
        df["is_high_value"] = (df["total_bet"] >= 1_000_000).astype(int)
    if "loss_chasing_score" not in df and {"loss_rate","total_sessions"}.issubset(df.columns):
        df["loss_chasing_score"] = (
            df["loss_rate"].clip(lower=0) * np.log1p(df["total_sessions"].clip(lower=0))
        )
    return df

def align_features_with_guard(df, expected_features, min_cov=0.95, abort_on_missing=True):
    """
    Align dataset features with model expectations and validate coverage.
    Ensures feature compatibility between training and prediction datasets.
    
    Args:
        df (pandas.DataFrame): Input customer data
        expected_features (list): Features expected by the trained model
        min_cov (float): Minimum feature coverage threshold (default: 0.95)
        abort_on_missing (bool): Whether to abort on missing features
    
    Returns:
        pandas.DataFrame: Aligned feature matrix ready for prediction
    """
    df = create_engineered_features(df.copy())

    present = [c for c in expected_features if c in df.columns]
    coverage = len(present) / max(1, len(expected_features))
    if coverage < min_cov:
        missing = [c for c in expected_features if c not in df.columns]
        msg = (f"[ABORT] Feature coverage {coverage:.2%} < {min_cov:.0%}. "
               f"Missing: {missing[:10]}{' ...' if len(missing)>10 else ''}")
        if abort_on_missing:
            raise ValueError(msg)
        else:
            print("[WARN]", msg)

    # Align features in expected order for model compatibility
    X = df.reindex(columns=expected_features)

    # Handle remaining missing values based on abort_on_missing setting
    if X.isna().any().any():
        if abort_on_missing:
            na_cols = X.columns[X.isna().any()].tolist()
            raise ValueError(f"[ABORT] Missing engineered columns not produced: {na_cols}")
        else:
            X = X.fillna(0.0)

    return X


# --- /PATCH ---
def match_features(data, expected_features):
    """
    Legacy feature matching function for backward compatibility.
    Note: This function is superseded by align_features_with_guard.
    """
    print("Matching features to model expectations...")
    
    available_features = []
    missing_features = []
    
    for feature in expected_features:
        if feature in data.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
    
    print(f"Available: {len(available_features)}/{len(expected_features)}")
    print(f"Available features: {available_features}")
    
    if missing_features:
        print(f"Missing features: {missing_features}")
        
        # Create missing features with zeros
        for missing_feature in missing_features:
            data[missing_feature] = 0
            available_features.append(missing_feature)
            print(f"Created missing feature {missing_feature} with zero values")
    
    # Return data with features in exact order expected by model
    X = data[expected_features].values
    
    print(f"Feature matrix shape: {X.shape}")
    
    return X, expected_features

def make_predictions(rf_model, scaler, label_encoder, X):
    """
    Generate predictions using trained Random Forest model.
    Applies scaling transformation and produces both class labels and probabilities.
    
    Args:
        rf_model: Trained Random Forest classifier
        scaler: Feature scaling transformer
        label_encoder: Class label encoder
        X: Feature matrix for prediction
    
    Returns:
        tuple: (predictions_labels, probabilities, predictions_numeric)
    """
    print("Generating predictions...")
    
    # Apply scaling
    try:
        X_scaled = scaler.transform(X)
        print("Feature scaling applied")
    except Exception as e:
        print(f"Warning: Scaling encountered error: {e}")
        X_scaled = X
    
    # Make predictions
    try:
        probabilities = rf_model.predict_proba(X_scaled)
        predictions_numeric = rf_model.predict(X_scaled)
        
        print(f"Predictions generated")
        print(f"   Probabilities shape: {probabilities.shape}")
        print(f"   Unique predictions: {np.unique(predictions_numeric)}")
        
        # Convert to labels if possible
        try:
            predictions_labels = label_encoder.inverse_transform(predictions_numeric)
            print(f"Converted to class labels: {np.unique(predictions_labels)}")
        except:
            predictions_labels = predictions_numeric
            print("Using numeric predictions")
        
        return predictions_labels, probabilities, predictions_numeric
        
    except Exception as e:
        print(f"Error: Prediction encountered error: {e}")
        raise

def analyze_and_save_results(customer_ids, predictions, probabilities, predictions_numeric):
    """
    Analyze prediction results and save to CSV file.
    Computes confidence statistics and prediction distributions for academic analysis.
    
    Args:
        customer_ids: Customer identifiers
        predictions: Predicted class labels
        probabilities: Prediction probability matrix
        predictions_numeric: Numeric class predictions
    
    Returns:
        tuple: (results_dataframe, average_confidence)
    """
    print("Analyzing prediction results...")
    
    # Basic analysis
    total_customers = len(predictions)
    confidence_scores = np.max(probabilities, axis=1)
    avg_confidence = confidence_scores.mean()
    
    print(f"Total customers analyzed: {total_customers:,}")
    print(f"Average prediction confidence: {avg_confidence:.3f}")
    
    # Prediction distribution
    unique_preds, counts = np.unique(predictions, return_counts=True)
    
    print(f"\nPrediction Distribution Analysis:")
    for pred, count in zip(unique_preds, counts):
        percentage = (count / total_customers) * 100
        avg_conf = confidence_scores[predictions == pred].mean()
        print(f"   {pred}: {count:,} customers ({percentage:.1f}%) - average confidence: {avg_conf:.3f}")
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'customer_id': customer_ids,
        'prediction': predictions,
        'prediction_numeric': predictions_numeric,
        'confidence': confidence_scores
    })
    
    # Add probability columns
    for i in range(probabilities.shape[1]):
        results_df[f'prob_class_{i}'] = probabilities[:, i]
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_file = f'working_predictions_{timestamp}.csv'
    results_df.to_csv(output_file, index=False)
    
    print(f"\nResults saved to: {output_file}")
    
    return results_df, avg_confidence

# Model/Period Arguments
def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--period", default="2023-H2")
    p.add_argument("--model-path", default=None,
                   help="Default: models/optimized_rf/optimized_kmeans_dbscan_rf_{period}.pkl")
    p.add_argument("--min-feature-coverage", type=float, default=0.95)

    # Boolean flag pair for abort-on-missing feature control
    p.add_argument("--abort-on-missing", dest="abort_on_missing", action="store_true", default=True)
    p.add_argument("--no-abort-on-missing", dest="abort_on_missing", action="store_false")
    return p.parse_args()



def main():
    """
    Main execution function for Random Forest prediction system.
    Orchestrates the complete prediction pipeline from model loading to result export.
    
    This function implements the academic research workflow for customer
    promotion prediction using pre-trained Random Forest models.
    """

    # STEP 0: Args
    args = parse_args()
    period = args.period
    model_path = args.model_path or f"/src/models/models/optimized_rf/optimized_kmeans_dbscan_rf_{period}.pkl"

    print("Random Forest Prediction System - University of Bath")
    print("=" * 50)
    print("Using validated Random Forest models for customer promotion prediction")

    try:
        # STEP 1: Load pre-trained Random Forest model
        print(f"\nSTEP 1: Loading Random Forest Model...")
        rf_model, scaler, label_encoder, feature_names = load_working_model(period)

        used_model_path = model_path  # Store for metadata

        # STEP 2: Load customer behavioral data
        print(f"\nSTEP 2: Loading Customer Data for {period}...")
        customer_data = load_customer_data(period)


        # STEP 3: Feature alignment and validation
        print(f"\nSTEP 3: Feature Alignment and Validation...")
        X = align_features_with_guard(
            customer_data,
            expected_features=feature_names,
            min_cov=args.min_feature_coverage,
            abort_on_missing=args.abort_on_missing
        )
        matched_features = list(X.columns)

        # STEP 4: Predict
        print(f"\nSTEP 4: Generating Predictions...")
        Xs = scaler.transform(X.values)
        proba = rf_model.predict_proba(Xs)
        yhat = rf_model.predict(Xs)
        labels = label_encoder.inverse_transform(yhat)

        # STEP 5: Analyze & Save
        print(f"\nSTEP 5: Saving Results...")
        ts = time.strftime("%Y%m%d_%H%M")
        out = customer_data[["customer_id"]].copy()
        out["period"] = period
        out["pred_label"] = labels
        out["pred_confidence"] = proba.max(axis=1)
        out["model_path"] = used_model_path
        out["n_expected_features"] = len(feature_names)
        out["n_features_present"] = X.shape[1]
        out["feature_coverage"] = float(X.shape[1]) / max(1, len(feature_names))
        out.to_csv(f"working_predictions_{period}_{ts}.csv", index=False)

        print(f"Analysis complete. Mean confidence: {out['pred_confidence'].mean():.3f}")
        print(f"Results saved to: working_predictions_{period}_{ts}.csv")

    except Exception as e:
        print("Error encountered:", e)
        raise



if __name__ == "__main__":
    results = main()