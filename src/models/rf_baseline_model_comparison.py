#!/usr/bin/env python3
"""
clean_harmonized_rf_training.py

University of Bath - Casino Research Project
Harmonized Random Forest Training Pipeline

Bu script:
1. customer_features + promo_label tablolarını birleştirir
2. Harmonized 4-class labeling sistemi ile Random Forest eğitir
3. Cross-validation ile model performansını ölçer
4. .pkl dosyası olarak kayıt eder
5. Academic compliance sağlar

Usage:
    python clean_harmonized_rf_training.py --period 2022-H2
    python clean_harmonized_rf_training.py --period 2023-H1
    python clean_harmonized_rf_training.py --period 2023-H2
"""

import pandas as pd
import numpy as np
import argparse
import logging
import joblib
import os
from datetime import datetime
from pathlib import Path

# Database connection
import psycopg2
from sqlalchemy import create_engine

# ML libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

# Warnings
import warnings
warnings.filterwarnings('ignore')

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'casino_research',
    'user': 'researcher',
    'password': 'academic_password_2024'
}

def setup_logging():
    """Configure logging for academic tracking"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('clean_harmonized_rf_training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def create_db_connection():
    """Create database connection"""
    try:
        engine = create_engine(
            f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        )
        return engine
    except Exception as e:
        raise Exception(f"Database connection failed: {e}")

def load_data(engine, period, logger):
    """
    Load and join customer_features + promo_label tables
    
    Returns:
        X: Feature matrix
        y: Target labels
        feature_names: List of feature column names
    """
    logger.info(f"Loading data for period: {period}")
    
    # Load customer features (casino_data schema) - period kolonu olmayabilir
    features_query = f"""
    SELECT * FROM casino_data.customer_features 
    ORDER BY customer_id
    """
    
    # Load promo labels (casino_data schema) - period filtresi
    labels_query = f"""
    SELECT customer_id, promo_label 
    FROM casino_data.promo_label 
    WHERE period = '{period}'
    ORDER BY customer_id
    """
    
    try:
        df_features = pd.read_sql(features_query, engine)
        df_labels = pd.read_sql(labels_query, engine)
        
        logger.info(f"Features loaded: {len(df_features)} rows")
        logger.info(f"Labels loaded: {len(df_labels)} rows")
        
        # Join on customer_id
        df_combined = df_features.merge(df_labels, on='customer_id', how='inner')
        logger.info(f"Combined dataset: {len(df_combined)} rows")
        
        # Prepare features (X) and target (y)
        feature_cols = [col for col in df_combined.columns 
                       if col not in ['customer_id', 'period', 'promo_label']]
        
        X = df_combined[feature_cols].copy()
        y = df_combined['promo_label'].copy()
        
        logger.info(f"Feature matrix shape: {X.shape}")
        logger.info(f"Target distribution:\n{y.value_counts()}")
        
        return X, y, feature_cols, df_combined
        
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise

def preprocess_data(X, y, logger):
    """
    Preprocess features and encode labels
    
    Returns:
        X_scaled: Standardized features
        y_encoded: Encoded labels
        scaler: Fitted StandardScaler
        label_encoder: Fitted LabelEncoder
    """
    logger.info("Starting data preprocessing...")
    
    # Handle missing values
    X_clean = X.fillna(X.median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    # Encode labels (harmonized 4-class system)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    logger.info(f"Label classes: {label_encoder.classes_}")
    logger.info(f"Encoded label distribution: {np.bincount(y_encoded)}")
    
    return X_scaled, y_encoded, scaler, label_encoder

def train_random_forest(X_train, X_test, y_train, y_test, logger, tune_hyperparams=False):
    """
    Train Random Forest with optional hyperparameter tuning
    
    Returns:
        rf_model: Trained RandomForestClassifier
        performance_metrics: Dictionary of performance metrics
    """
    logger.info("Training Random Forest model...")
    
    # Calculate class weights for imbalanced data
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    logger.info(f"Class weights: {class_weight_dict}")
    
    if tune_hyperparams:
        logger.info("Performing hyperparameter tuning...")
        
        # Grid search parameters
        param_grid = {
            'n_estimators': [100, 150, 200],
            'max_depth': [8, 10, 12],
            'min_samples_split': [5, 8, 10],
            'min_samples_leaf': [2, 4, 6]
        }
        
        rf = RandomForestClassifier(
            random_state=42,
            class_weight=class_weight_dict,
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            rf, param_grid, 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        rf_model = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
    else:
        # Standard Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=150,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            random_state=42,
            class_weight=class_weight_dict,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf_model.predict(X_test)
    y_pred_proba = rf_model.predict_proba(X_test)
    
    # Performance metrics
    accuracy = rf_model.score(X_test, y_test)
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    # ROC AUC (multi-class)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    except:
        roc_auc = None
        logger.warning("ROC AUC calculation failed - likely due to missing classes in test set")
    
    performance_metrics = {
        'accuracy': accuracy,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'roc_auc': roc_auc,
        'cv_scores': cv_scores
    }
    
    logger.info(f"Model Performance:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  CV Mean: {cv_mean:.4f} ± {cv_std:.4f}")
    if roc_auc:
        logger.info(f"  ROC AUC: {roc_auc:.4f}")
    
    # Classification report
    logger.info(f"\nClassification Report:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    
    return rf_model, performance_metrics

def save_model(rf_model, scaler, label_encoder, performance_metrics, 
               feature_names, period, logger):
    """
    Save trained model and metadata
    
    Returns:
        model_filename: Path to saved model file
    """
    # Create models directory
    models_dir = Path("models/generic_rf")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_filename = f"clean_harmonized_rf_{period}_v1_{timestamp}.pkl"
    model_path = models_dir / model_filename
    
    # Model metadata
    model_metadata = {
        'model': rf_model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'feature_names': feature_names,
        'period': period,
        'performance': performance_metrics,
        'training_timestamp': timestamp,
        'model_type': 'RandomForestClassifier',
        'harmonized_labels': True,
        'n_classes': len(label_encoder.classes_),
        'classes': label_encoder.classes_.tolist()
    }
    
    # Save model
    joblib.dump(model_metadata, model_path)
    logger.info(f"Model saved: {model_path}")
    
    # Update model registry
    update_model_registry(model_filename, period, performance_metrics, logger)
    
    # Create LATEST.pkl symlink
    latest_path = models_dir / "LATEST.pkl"
    if latest_path.exists():
        latest_path.unlink()
    
    # Create relative symlink
    os.symlink(model_filename, latest_path)
    logger.info(f"LATEST.pkl updated: {model_filename}")
    
    return model_path

def update_model_registry(model_filename, period, performance_metrics, logger):
    """Update model registry log"""
    registry_file = "model_registry.log"
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    registry_entry = (
        f"{timestamp} | {model_filename} | {period} | "
        f"Accuracy: {performance_metrics['accuracy']:.4f} | "
        f"CV: {performance_metrics['cv_mean']:.4f}±{performance_metrics['cv_std']:.4f} | "
        f"ROC_AUC: {performance_metrics.get('roc_auc', 'N/A')} | "
        f"Harmonized: True\n"
    )
    
    with open(registry_file, 'a') as f:
        f.write(registry_entry)
    
    logger.info(f"Model registry updated: {registry_file}")

def generate_feature_importance_report(rf_model, feature_names, period, logger):
    """Generate feature importance analysis"""
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"\nTop 10 Feature Importances for {period}:")
    logger.info(importance_df.head(10).to_string(index=False))
    
    # Save to CSV
    importance_file = f"feature_importance_{period}.csv"
    importance_df.to_csv(importance_file, index=False)
    logger.info(f"Feature importance saved: {importance_file}")
    
    return importance_df

def main():
    # Command line arguments
    parser = argparse.ArgumentParser(description='Clean Harmonized Random Forest Training')
    parser.add_argument('--period', required=True, 
                       choices=['2022-H2', '2023-H1', '2023-H2'],
                       help='Training period')
    parser.add_argument('--tune', action='store_true',
                       help='Enable hyperparameter tuning')
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    logger.info(f"Starting harmonized RF training for period: {args.period}")
    logger.info(f"Hyperparameter tuning: {'Enabled' if args.tune else 'Disabled'}")
    
    try:
        # Database connection
        engine = create_db_connection()
        logger.info("Database connection established")
        
        # Load data
        X, y, feature_names, df_combined = load_data(engine, args.period, logger)
        
        # Preprocess data
        X_scaled, y_encoded, scaler, label_encoder = preprocess_data(X, y, logger)
        
        # Train-test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded,
            test_size=0.2,
            random_state=42,
            stratify=y_encoded
        )
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        
        # Train model
        rf_model, performance_metrics = train_random_forest(
            X_train, X_test, y_train, y_test, 
            logger, tune_hyperparams=args.tune
        )
        
        # Save model
        model_path = save_model(
            rf_model, scaler, label_encoder, performance_metrics,
            feature_names, args.period, logger
        )
        
        # Feature importance analysis
        importance_df = generate_feature_importance_report(
            rf_model, feature_names, args.period, logger
        )
        
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Accuracy: {performance_metrics['accuracy']:.4f}")
        logger.info(f"CV Score: {performance_metrics['cv_mean']:.4f} ± {performance_metrics['cv_std']:.4f}")
        
        if performance_metrics['roc_auc']:
            logger.info(f"ROC AUC: {performance_metrics['roc_auc']:.4f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()