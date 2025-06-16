# main_pipeline.py

"""
Main Pipeline for Casino Customer Segmentation
==============================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

This script orchestrates the complete ML pipeline from data to predictions.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

from src.config.academic_config import ACADEMIC_METADATA, get_academic_header
from src.data.db_connector import AcademicDBConnector
from src.data.anonymizer import AcademicDataAnonymizer
from src.data.compatibility_layer import SyntheticToRealMapper
from src.features.feature_engineering import CasinoFeatureEngineer
from src.features.temporal_features import TemporalFeatureEngineer
from src.models.segmentation import CustomerSegmentation
from src.models.promotion_rf import PromotionResponseModel
from src.models.model_registry import ModelRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CasinoPipeline:
    """
    Main pipeline orchestrator for thesis project.
    Handles data flows from DB to trained models.
    """
    
    def __init__(self, mode: str = "synthetic"):
        """
        Initialize pipeline.
        
        Args:
            mode: 'synthetic' for CSV data or 'database' for PostgreSQL
        """
        self.mode = mode
        self.db_connector = AcademicDBConnector() if mode == "database" else None
        self.anonymizer = AcademicDataAnonymizer()
        self.mapper = SyntheticToRealMapper()
        self.feature_engineer = CasinoFeatureEngineer()
        self.temporal_engineer = TemporalFeatureEngineer()
        self.segmentation_model = CustomerSegmentation()
        self.promotion_model = PromotionResponseModel()
        self.model_registry = ModelRegistry()
        
        logger.info(f"Pipeline initialized in {mode} mode")
        logger.info(f"Ethics approval: {ACADEMIC_METADATA['ethics_ref']}")
    
    def load_synthetic_data(self) -> pd.DataFrame:
        """Load synthetic data from Casino-1 project"""
        logger.info("Loading synthetic data...")
        
        # For now, create sample data
        # In production, load from CSV files
        np.random.seed(42)
        n_customers = 1000
        
        data = {
            'customer_id': [f'CUST_{str(i).zfill(6)}' for i in range(n_customers)],
            'age': np.random.randint(18, 80, n_customers),
            'gender': np.random.choice(['M', 'F'], n_customers),
            'location': np.random.choice(['Sofia', 'Plovdiv', 'Varna'], n_customers),
            'total_sessions': np.random.randint(1, 100, n_customers),
            'total_bet': np.random.exponential(1000, n_customers),
            'total_win': np.random.exponential(900, n_customers),
            'avg_session_duration': np.random.normal(45, 15, n_customers),
            'days_since_last_visit': np.random.randint(0, 90, n_customers),
        }
        
        df = pd.DataFrame(data)
        
        # Create session-level data
        sessions = []
        for _, customer in df.iterrows():
            for _ in range(int(customer['total_sessions'])):
                session = {
                    'customer_id': customer['customer_id'],
                    'session_id': np.random.randint(10000, 99999),
                    'start_time': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 90)),
                    'end_time': pd.Timestamp.now() - pd.Timedelta(days=np.random.randint(0, 90)),
                    'total_bet': customer['total_bet'] / customer['total_sessions'],
                    'total_win': customer['total_win'] / customer['total_sessions'],
                    'game_type': np.random.choice(['Slots', 'Roulette', 'Blackjack']),
                    'machine_id': f'MACHINE_{np.random.randint(1, 50)}'
                }
                sessions.append(session)
        
        sessions_df = pd.DataFrame(sessions)
        
        logger.info(f"Loaded {len(df)} customers with {len(sessions_df)} sessions")
        return df, sessions_df
    
    def load_database_data(self) -> pd.DataFrame:
        """Load data from PostgreSQL database"""
        logger.info("Loading data from database...")
        
        # Query to get customer data
        query = """
        SELECT 
            c.customer_id,
            c.age_range,
            c.gender,
            c.region,
            COUNT(DISTINCT s.session_id) as total_sessions,
            SUM(s.total_bet) as total_wagered,
            SUM(s.total_win) as total_winnings
        FROM casino_data.customer_demographics c
        LEFT JOIN casino_data.player_sessions s ON c.customer_id = s.customer_id
        GROUP BY c.customer_id, c.age_range, c.gender, c.region
        """
        
        try:
            with self.db_connector.get_connection() as conn:
                df = pd.read_sql(query, conn)
                logger.info(f"Loaded {len(df)} customers from database")
                return df
        except Exception as e:
            logger.error(f"Database load failed: {e}")
            logger.info("Falling back to synthetic data")
            return self.load_synthetic_data()
    
    def prepare_features(self, customer_df: pd.DataFrame, 
                        sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for ML"""
        logger.info("Engineering features...")
        
        # Basic features
        basic_features = self.feature_engineer.create_basic_features(sessions_df)
        
        # Temporal features
        temporal_features = self.temporal_engineer.create_temporal_feature_matrix(sessions_df)
        
        # Behavioral features
        behavioral_features = self.feature_engineer.create_behavioral_features(sessions_df)
        
        # Merge all features
        feature_matrix = basic_features.merge(
            temporal_features, on='customer_id', how='left'
        ).merge(
            behavioral_features, on='customer_id', how='left'
        )
        
        # Fill missing values
        feature_matrix = feature_matrix.fillna(0)
        
        logger.info(f"Created feature matrix with shape: {feature_matrix.shape}")
        return feature_matrix
    
    def train_models(self, feature_matrix: pd.DataFrame) -> Tuple[Dict, Dict]:
        """Train segmentation and promotion models"""
        logger.info("Training models...")
        
        # Train segmentation model
        logger.info("Training segmentation model...")
        self.segmentation_model.fit(feature_matrix)
        
        # Add segments to features
        feature_matrix['segment'] = self.segmentation_model.predict(feature_matrix)
        
        # Create synthetic labels for promotion model (in real scenario, use historical data)
        # High-value customers and at-risk players more likely to respond
        promo_probability = (
            (feature_matrix['total_wagered'] > feature_matrix['total_wagered'].median()).astype(float) * 0.3 +
            (feature_matrix['segment'].isin([1, 3])).astype(float) * 0.4 +
            np.random.random(len(feature_matrix)) * 0.3
        )
        y_promo = (promo_probability > 0.5).astype(int)
        
        # Train promotion model
        logger.info("Training promotion model...")
        self.promotion_model.fit(feature_matrix, y_promo)
        
        # Get performance metrics
        segmentation_metrics = {
            'n_clusters': self.segmentation_model.n_clusters,
            'silhouette_score': self.segmentation_model.model_metadata.get('silhouette_score', 0),
            'davies_bouldin_score': self.segmentation_model.model_metadata.get('davies_bouldin_score', 0)
        }
        
        promotion_metrics = self.promotion_model.evaluate(feature_matrix, y_promo)
        
        return segmentation_metrics, promotion_metrics
    
    def save_models(self):
        """Save trained models and register them"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save segmentation model
        seg_path = f'models/segmentation_{timestamp}.pkl'
        os.makedirs('models', exist_ok=True)
        self.segmentation_model.save_model(seg_path)
        
        # Register segmentation model
        seg_id = self.model_registry.register_model(
            model_type='segmentation',
            model_path=seg_path,
            metrics=self.segmentation_model.model_metadata,
            notes='Academic thesis model - customer segmentation'
        )
        
        # Save promotion model
        promo_path = f'models/promotion_{timestamp}.pkl'
        self.promotion_model.save_model(promo_path)
        
        # Register promotion model
        promo_id = self.model_registry.register_model(
            model_type='promotion',
            model_path=promo_path,
            metrics=self.promotion_model.model_metadata,
            notes='Academic thesis model - promotion response'
        )
        
        logger.info(f"Models saved and registered: {seg_id}, {promo_id}")
    
    def run_pipeline(self):
        """Execute complete pipeline"""
        logger.info("="*50)
        logger.info("STARTING CASINO ML PIPELINE")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Ethics Approval: {ACADEMIC_METADATA['ethics_ref']}")
        logger.info("="*50)
        
        # Step 1: Load data
        if self.mode == "synthetic":
            customer_df, sessions_df = self.load_synthetic_data()
        else:
            customer_df = self.load_database_data()
            # In real scenario, load sessions separately
            sessions_df = None
        
        # Step 2: Prepare features
        feature_matrix = self.prepare_features(customer_df, sessions_df)
        
        # Step 3: Train models
        seg_metrics, promo_metrics = self.train_models(feature_matrix)
        
        # Step 4: Display results
        logger.info("\n" + "="*50)
        logger.info("PIPELINE RESULTS")
        logger.info("="*50)
        
        logger.info("\nSegmentation Model Performance:")
        for key, value in seg_metrics.items():
            logger.info(f"  {key}: {value:.3f}")
        
        logger.info("\nSegment Summary:")
        segment_summary = self.segmentation_model.get_segment_summary()
        print(segment_summary)
        
        logger.info("\nPromotion Model Performance:")
        logger.info(f"  ROC-AUC: {promo_metrics['roc_auc']:.3f}")
        logger.info(f"  Accuracy: {promo_metrics['accuracy']:.3f}")
        logger.info(f"  Precision: {promo_metrics['precision_positive']:.3f}")
        logger.info(f"  Recall: {promo_metrics['recall_positive']:.3f}")
        
        logger.info("\nTop Feature Importance:")
        feature_importance = self.promotion_model.get_feature_importance(top_n=5)
        print(feature_importance)
        
        # Step 5: Save models
        self.save_models()
        
        logger.info("\n" + "="*50)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*50)

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Casino Customer Segmentation ML Pipeline'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='synthetic',
        choices=['synthetic', 'database'],
        help='Data source mode'
    )
    
    args = parser.parse_args()
    
    # Print academic header
    print(get_academic_header("CASINO CUSTOMER SEGMENTATION PIPELINE"))
    
    # Run pipeline
    pipeline = CasinoPipeline(mode=args.mode)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()