"""
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

For 2024-H1 predictions - Test Environment
This module implements an advanced Random Forest training pipeline specifically designed
to address severe class imbalance issues in casino customer promotion response prediction.
The implementation incorporates SMOTE oversampling and random undersampling techniques
to achieve balanced multi-class prediction capabilities.
"""

import os
import json
import datetime as dt
import numpy as np
import pandas as pd

from pathlib import Path
from joblib import parallel_config
from sqlalchemy import create_engine, text
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'casino_research',
    'user': 'researcher',
    'password': 'academic_password_2024'
}
DB_URL = f"postgresql+psycopg2://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:5432/{DB_CONFIG['database']}"
engine = create_engine(DB_URL, pool_pre_ping=True)

# Configuration constants
TRAIN_PERIODS = ('2022-H1','2022-H2','2023-H1','2023-H2')
PREDICT_PERIOD = '2024-H1'
FEATURES = [
    "total_events","total_bet","avg_bet","bet_std","total_win","avg_win","avg_loss","loss_rate",
    "total_sessions","avg_events_per_session","game_diversity","multi_game_player","machine_diversity","zone_diversity",
    "bet_volatility","weekend_preference","late_night_player","days_since_last_visit",
    "session_duration_volatility","loss_chasing_score","sessions_last_30d","bet_trend_ratio"
]
HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "models" / "generic_rf"
OUTDIR.mkdir(parents=True, exist_ok=True)

def load_train():
    sql = text(f"""
        SELECT
            cf.customer_id,
            cf.analysis_period AS period,
            {", ".join("cf."+c for c in FEATURES)},
            pl.promo_label
        FROM casino_data.customer_features cf
        JOIN casino_data.promo_label pl
          ON pl.customer_id = cf.customer_id
         AND pl.period      = cf.analysis_period
        WHERE cf.analysis_period = ANY(:periods)
    """)
    df = pd.read_sql(sql, engine, params={"periods": list(TRAIN_PERIODS)})
    if df.empty:
        raise RuntimeError("No labeled training data found for 2022-2023 periods.")
    # Remove rows with missing critical values
    df = df.dropna(subset=["total_bet","loss_rate","loss_chasing_score"])
    # Fill remaining missing values with 0
    df[FEATURES] = df[FEATURES].fillna(0)
    X = df[FEATURES].astype(float).values
    y = df["promo_label"].astype(str).values
    return X, y

def train_rf(X, y):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,  # Random Forest uses internal threading
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Use THREADS backend for Windows compatibility
    with parallel_config(prefer="threads"):
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=-1)
        pipe.fit(X, y)

    perf = {
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "cv_scores": [float(s) for s in scores],
    }
    return pipe, perf

def load_predict_df():
    # Load 2024-H1 data from validated view
    sql = text("""
        SELECT customer_id, analysis_period as period,
               total_events,total_bet,avg_bet,bet_std,total_win,avg_win,avg_loss,loss_rate,
               total_sessions,avg_events_per_session,game_diversity,multi_game_player,machine_diversity,zone_diversity,
               bet_volatility,weekend_preference,late_night_player,days_since_last_visit,
               session_duration_volatility,loss_chasing_score,sessions_last_30d,bet_trend_ratio
        FROM casino_data.kmeans_export_2024_h1
        ORDER BY customer_id
    """)
    df = pd.read_sql(sql, engine)
    if df.empty:
        raise RuntimeError("2024-H1 view is empty.")
    df[FEATURES] = df[FEATURES].fillna(0)
    return df

def predict_and_upsert(model, df_pred):
    Xp = df_pred[FEATURES].astype(float).values
    preds = model.predict(Xp)
    proba = model.predict_proba(Xp)
    conf = proba.max(axis=1)

    out = pd.DataFrame({
        "customer_id": df_pred["customer_id"].values,
        "period": df_pred["period"].values,            # '2024-H1'
        "promo_label": preds.astype(str),
        "label_confidence": np.round(conf, 3),
        "risk_score": np.zeros(len(conf), dtype=float),   # Default to 0 if no business scores
        "value_score": np.zeros(len(conf), dtype=float)
    })

    with engine.begin() as con:
        # Ensure table exists
        con.execute(text("""
            CREATE TABLE IF NOT EXISTS casino_data.promo_label(
              customer_id BIGINT,
              period TEXT,
              promo_label TEXT,
              label_confidence NUMERIC,
              risk_score NUMERIC,
              value_score NUMERIC,
              PRIMARY KEY (customer_id, period)
            );
        """))
        # Write to temporary table
        out.to_sql("_tmp_pred_2024h1", con, schema="casino_data", if_exists="replace", index=False)
        # Perform upsert operation
        con.execute(text("""
            INSERT INTO casino_data.promo_label AS t
              (customer_id, period, promo_label, label_confidence, risk_score, value_score)
            SELECT customer_id, period, promo_label, label_confidence, risk_score, value_score
            FROM casino_data._tmp_pred_2024h1
            ON CONFLICT (customer_id, period) DO UPDATE
            SET promo_label = EXCLUDED.promo_label,
                label_confidence = EXCLUDED.label_confidence,
                risk_score = EXCLUDED.risk_score,
                value_score = EXCLUDED.value_score;
            DROP TABLE casino_data._tmp_pred_2024h1;
        """))
    return out

def main():
    print("[INFO] DB_URL:", DB_URL)
    print("[1/4] Loading training data (2022-2023)...")
    X, y = load_train()
    print(f"      -> {X.shape[0]} rows, {X.shape[1]} features")

    print("[2/4] Training model...")
    model, perf = train_rf(X, y)
    print(f"      -> CV acc: {perf['cv_mean']:.4f} ± {perf['cv_std']:.4f}")

    meta = {
        "features": FEATURES,
        "trained_periods": TRAIN_PERIODS,
        "predict_period": PREDICT_PERIOD,
        "performance": perf,
        "ts": dt.datetime.now().strftime("%Y%m%d_%H%M")
    }
    pkl_path = os.path.join(OUTDIR, f"clean_harmonized_rf_22_23_to_{PREDICT_PERIOD}_{meta['ts']}.pkl")
    import joblib
    joblib.dump({"model": model, "features": FEATURES, "meta": meta}, pkl_path)
    with open(pkl_path + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[3/4] Model saved -> {pkl_path}")

    print("[4/4] 2024-H1 prediction + upsert...")
    dfp = load_predict_df()
    out = predict_and_upsert(model, dfp)
    print(f"      -> rows written: {len(out)}")

    # Quick validation
    with engine.connect() as con:
        c = con.execute(text("SELECT COUNT(*) FROM casino_data.promo_label WHERE period=:p"), {"p": PREDICT_PERIOD}).scalar()
        print(f"[OK] promo_label({PREDICT_PERIOD}) count = {c}")
        
    # Post-training validation
    classes_ = np.unique(y)
    print("[CHECK] Classes used in training:", list(map(str, classes_)))

if __name__ == "__main__":
    main()
