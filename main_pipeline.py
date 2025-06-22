# main_pipeline.py

"""
Main Pipeline for Casino Customer Segmentation
==============================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

This script orchestrates the complete ML pipeline from data to predictions.
Supports multiple data sources for academic comparison.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.abspath('src'))

from src.config.academic_config import ACADEMIC_METADATA, get_academic_header
from src.data.db_connector import AcademicDBConnector
from src.data.anonymizer import AcademicDataAnonymizer
#from src.data.compatibility_layer import SyntheticToRealMapper - will return back !!!
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
    
    Supports three modes:
    1. synthetic: In-memory generated data (Casino-1 enhanced)
    2. batch: Real data from PostgreSQL (Casino-2)
    3. comparison: Compare both approaches
    
    Academic Note:
        This multi-mode approach allows validation of synthetic-to-real
        data transition, a key contribution of this thesis.
    """
    
    def __init__(self, mode: str = "synthetic"):
        """
        Initialize pipeline.
        
        Args:
            mode: 'synthetic' (Casino-1), 'batch' (Casino-2), or 'comparison'
        """
        self.mode = mode
        self.db_connector = AcademicDBConnector() if mode in ["batch", "comparison"] else None
        self.anonymizer = AcademicDataAnonymizer()
        # self.mapper = SyntheticToRealMapper() -- will return back !!!
        self.feature_engineer = CasinoFeatureEngineer()
        self.temporal_engineer = TemporalFeatureEngineer()
        self.segmentation_model = CustomerSegmentation()
        self.promotion_model = PromotionResponseModel()
        self.model_registry = ModelRegistry()
        
        logger.info(f"Pipeline initialized in {mode} mode")
        logger.info(f"Ethics approval: {ACADEMIC_METADATA['ethics_ref']}")
    
    def load_synthetic_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load synthetic data (Casino-1 enhanced version).
        
        Academic Note:
            This synthetic data generation maintains the same statistical
            properties as expected from real casino data, allowing for
            controlled experimentation.
            
        Returns:
            Tuple of (customer_df, sessions_df)
        """
        logger.info("Loading synthetic data (Casino-1 mode)...")
        
        # Keep existing synthetic data generation
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
    
    def load_batch_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load real batch data from PostgreSQL (Casino-2).
        
        Academic Note:
            This loads pre-anonymized data provided by Casino IT department,
            maintaining full compliance with ethics approval 10351-12382.
            
        Returns:
            Tuple of (customer_df, sessions_df)
        """
        logger.info("Loading real batch data (Casino-2 mode)...")
        
        try:
            with self.db_connector.get_connection() as conn:
                # Get latest import session
                latest_session_query = """
                SELECT MAX(import_session_id) as latest_session
                FROM casino_data.customer_demographics
                """
                latest_session = pd.read_sql(latest_session_query, conn).iloc[0]['latest_session']
                
                logger.info(f"Loading data from import session: {latest_session}")
                
                # Customer demographics
                customers_query = f"""
                SELECT 
                    customer_id,
                    age_range,
                    gender,
                    region,
                    registration_month
                FROM casino_data.customer_demographics
                WHERE import_session_id = '{latest_session}'
                """
                customers_df = pd.read_sql(customers_query, conn)
                
                # Player sessions
                sessions_query = f"""
                SELECT 
                    session_id,
                    customer_id,
                    session_start as start_time,
                    session_end as end_time,
                    total_bet,
                    total_win,
                    game_type,
                    machine_id
                FROM casino_data.player_sessions
                WHERE import_session_id = '{latest_session}'
                """
                sessions_df = pd.read_sql(sessions_query, conn)
                
                logger.info(f"Loaded {len(customers_df)} customers, {len(sessions_df)} sessions from batch")
                
                # Log to audit
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO academic_audit.access_log 
                        (student_id, ethics_ref, action, table_accessed, 
                         query_type, record_count, timestamp)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        ACADEMIC_METADATA['student_id'],
                        ACADEMIC_METADATA['ethics_ref'],
                        'BATCH_DATA_LOAD',
                        'customer_demographics,player_sessions',
                        'SELECT',
                        len(customers_df) + len(sessions_df),
                        datetime.now()
                    ))
                    conn.commit()
                
                return customers_df, sessions_df
                
        except Exception as e:
            logger.error(f"Batch data load failed: {e}")
            logger.info("Falling back to synthetic data")
            return self.load_synthetic_data()
    
    def load_database_data(self) -> pd.DataFrame:
        """Legacy method for backward compatibility"""
        logger.warning("load_database_data is deprecated, use load_batch_data instead")
        customers_df, sessions_df = self.load_batch_data()
        return customers_df
    
    def prepare_features(self, customer_df: pd.DataFrame, 
                        sessions_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare feature matrix for ML - unchanged from original"""
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
    
    def train_models(self, feature_matrix: pd.DataFrame, 
                    label_source: str = "synthetic") -> Tuple[Dict, Dict]:
        """
        Train segmentation and promotion models.
        
        Args:
            feature_matrix: Prepared features
            label_source: 'synthetic' or 'historical' for promotion labels
        """
        logger.info(f"Training models with {label_source} labels...")
        
        # Train segmentation model
        logger.info("Training segmentation model...")
        self.segmentation_model.fit(feature_matrix)
        
        # Add segments to features
        feature_matrix['segment'] = self.segmentation_model.predict(feature_matrix)
        
        # Get promotion labels
        if label_source == "synthetic":
            # Create synthetic labels (existing logic)
            promo_probability = (
                (feature_matrix['total_wagered'] > feature_matrix['total_wagered'].median()).astype(float) * 0.3 +
                (feature_matrix['segment'].isin([1, 3])).astype(float) * 0.4 +
                np.random.random(len(feature_matrix)) * 0.3
            )
            y_promo = (promo_probability > 0.5).astype(int)
            
        elif label_source == "historical":
            # Load historical promotion responses from database
            y_promo = self._load_historical_promotion_labels(feature_matrix['customer_id'])
            
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
    
    def _load_historical_promotion_labels(self, customer_ids: pd.Series) -> np.ndarray:
        """Load historical promotion responses from database"""
        try:
            with self.db_connector.get_connection() as conn:
                # Get historical responses
                query = """
                SELECT 
                    customer_id,
                    MAX(CASE WHEN response = true THEN 1 ELSE 0 END) as responded
                FROM casino_data.promotion_history
                WHERE customer_id = ANY(%s)
                GROUP BY customer_id
                """
                responses = pd.read_sql(query, conn, params=(list(customer_ids),))
                
                # Merge with customer list
                merged = pd.DataFrame({'customer_id': customer_ids}).merge(
                    responses, on='customer_id', how='left'
                )
                
                # Fill missing with 0 (no response)
                return merged['responded'].fillna(0).values
                
        except Exception as e:
            logger.warning(f"Could not load historical labels: {e}")
            logger.info("Falling back to synthetic labels")
            return None
    
    def run_comparison_analysis(self):
        """
        Compare Casino-1 (synthetic) vs Casino-2 (real batch) results.
        
        Academic contribution: Validates model transferability from
        synthetic to real data environments.
        """
        logger.info("\n" + "="*70)
        logger.info("COMPARATIVE ANALYSIS: CASINO-1 vs CASINO-2")
        logger.info("="*70)
        
        comparison_results = {}
        
        # 1. Train on synthetic data (Casino-1)
        logger.info("\n### CASINO-1 (Synthetic Data) ###")
        synthetic_start = datetime.now()
        
        synthetic_customers, synthetic_sessions = self.load_synthetic_data()
        synthetic_features = self.prepare_features(synthetic_customers, synthetic_sessions)
        synthetic_seg_metrics, synthetic_promo_metrics = self.train_models(
            synthetic_features, label_source="synthetic"
        )
        
        synthetic_time = (datetime.now() - synthetic_start).total_seconds()
        
        comparison_results['casino1'] = {
            'data_size': len(synthetic_customers),
            'session_count': len(synthetic_sessions),
            'feature_count': synthetic_features.shape[1],
            'segmentation_metrics': synthetic_seg_metrics,
            'promotion_metrics': synthetic_promo_metrics,
            'training_time': synthetic_time,
            'segment_distribution': self.segmentation_model.get_segment_summary().to_dict()
        }
        
        # 2. Train on real data (Casino-2)
        logger.info("\n### CASINO-2 (Real Batch Data) ###")
        real_start = datetime.now()
        
        real_customers, real_sessions = self.load_batch_data()
        real_features = self.prepare_features(real_customers, real_sessions)
        
        # Check if historical labels available
        label_source = "historical" if self._check_historical_data_available() else "synthetic"
        real_seg_metrics, real_promo_metrics = self.train_models(
            real_features, label_source=label_source
        )
        
        real_time = (datetime.now() - real_start).total_seconds()
        
        comparison_results['casino2'] = {
            'data_size': len(real_customers),
            'session_count': len(real_sessions),
            'feature_count': real_features.shape[1],
            'segmentation_metrics': real_seg_metrics,
            'promotion_metrics': real_promo_metrics,
            'training_time': real_time,
            'segment_distribution': self.segmentation_model.get_segment_summary().to_dict(),
            'label_source': label_source
        }
        
        # 3. Generate comparison report
        self._generate_comparison_report(comparison_results)
        
        # 4. Save comparison results
        comparison_file = f'comparison_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        import json
        with open(comparison_file, 'w') as f:
            json.dump(comparison_results, f, indent=2, default=str)
        
        logger.info(f"\nComparison results saved to: {comparison_file}")
    
    def _check_historical_data_available(self) -> bool:
        """Check if historical promotion data is available"""
        try:
            with self.db_connector.get_connection() as conn:
                query = "SELECT COUNT(*) FROM casino_data.promotion_history"
                count = pd.read_sql(query, conn).iloc[0, 0]
                return count > 0
        except:
            return False
    
    def _generate_comparison_report(self, results: Dict):
        """Generate detailed comparison report"""
        logger.info("\n" + "="*70)
        logger.info("COMPARISON REPORT")
        logger.info("="*70)
        
        # Data characteristics
        logger.info("\n### Data Characteristics ###")
        logger.info(f"Casino-1 Customers: {results['casino1']['data_size']:,}")
        logger.info(f"Casino-2 Customers: {results['casino2']['data_size']:,}")
        logger.info(f"Casino-1 Sessions: {results['casino1']['session_count']:,}")
        logger.info(f"Casino-2 Sessions: {results['casino2']['session_count']:,}")
        
        # Segmentation comparison
        logger.info("\n### Segmentation Performance ###")
        logger.info(f"Casino-1 Silhouette: {results['casino1']['segmentation_metrics']['silhouette_score']:.3f}")
        logger.info(f"Casino-2 Silhouette: {results['casino2']['segmentation_metrics']['silhouette_score']:.3f}")
        
        # Promotion model comparison
        logger.info("\n### Promotion Model Performance ###")
        logger.info(f"Casino-1 ROC-AUC: {results['casino1']['promotion_metrics']['roc_auc']:.3f}")
        logger.info(f"Casino-2 ROC-AUC: {results['casino2']['promotion_metrics']['roc_auc']:.3f}")
        logger.info(f"Casino-2 Label Source: {results['casino2'].get('label_source', 'synthetic')}")
        
        # Performance comparison
        logger.info("\n### Performance Metrics ###")
        logger.info(f"Casino-1 Training Time: {results['casino1']['training_time']:.2f}s")
        logger.info(f"Casino-2 Training Time: {results['casino2']['training_time']:.2f}s")
        
        # Key insights
        logger.info("\n### Key Insights ###")
        silhouette_diff = results['casino2']['segmentation_metrics']['silhouette_score'] - \
                         results['casino1']['segmentation_metrics']['silhouette_score']
        
        if abs(silhouette_diff) < 0.05:
            logger.info("✓ Segmentation quality is consistent between synthetic and real data")
        else:
            logger.info(f"⚠ Segmentation quality differs by {silhouette_diff:.3f}")
        
        roc_diff = results['casino2']['promotion_metrics']['roc_auc'] - \
                   results['casino1']['promotion_metrics']['roc_auc']
        
        if abs(roc_diff) < 0.1:
            logger.info("✓ Promotion model performance is stable across data sources")
        else:
            logger.info(f"⚠ Promotion model performance differs by {roc_diff:.3f}")
    
    def save_models(self):
        """Save trained models and register them - unchanged from original"""
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
            notes=f'Academic thesis model - {self.mode} mode'
        )
        
        # Save promotion model
        promo_path = f'models/promotion_{timestamp}.pkl'
        self.promotion_model.save_model(promo_path)
        
        # Register promotion model
        promo_id = self.model_registry.register_model(
            model_type='promotion',
            model_path=promo_path,
            metrics=self.promotion_model.model_metadata,
            notes=f'Academic thesis model - {self.mode} mode'
        )
        
        logger.info(f"Models saved and registered: {seg_id}, {promo_id}")
    
    def run_pipeline(self):
        """Execute complete pipeline"""
        logger.info("="*50)
        logger.info("STARTING CASINO ML PIPELINE")
        logger.info(f"Mode: {self.mode}")
        logger.info(f"Ethics Approval: {ACADEMIC_METADATA['ethics_ref']}")
        logger.info("="*50)
        
        # Handle comparison mode separately
        if self.mode == "comparison":
            self.run_comparison_analysis()
            return
        
        # Step 1: Load data
        if self.mode == "synthetic":
            customer_df, sessions_df = self.load_synthetic_data()
            label_source = "synthetic"
        elif self.mode == "batch":
            customer_df, sessions_df = self.load_batch_data()
            label_source = "historical" if self._check_historical_data_available() else "synthetic"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
        
        # Step 2: Prepare features
        feature_matrix = self.prepare_features(customer_df, sessions_df)
        
        # Step 3: Train models
        seg_metrics, promo_metrics = self.train_models(feature_matrix, label_source)
        
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
        logger.info(f"  Label Source: {label_source}")
        
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
        description='Casino Customer Segmentation AI-ML Pipeline'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='synthetic',
        choices=['synthetic', 'batch', 'comparison'],
        help='Data source mode: synthetic (Casino-1), batch (Casino-2), or comparison'
    )
    
    args = parser.parse_args()
    
    # Print academic header
    print(get_academic_header("CASINO CUSTOMER SEGMENTATION PIPELINE"))
    
    # Run pipeline
    pipeline = CasinoPipeline(mode=args.mode)
    pipeline.run_pipeline()

if __name__ == "__main__":
    main()