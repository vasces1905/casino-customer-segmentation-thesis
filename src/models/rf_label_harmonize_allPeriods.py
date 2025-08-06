#!/usr/bin/env python3
"""
Label Harmonizer for All Periods - University of Bath Academic Standard
======================================================================
Based on the problem I have faced:
- The LabelEncoder in the .pkl file contains only the labels it was trained on at that time.
Example:
- During the training in 2022-H1 there were the following classes as target_label:
['NO_PROMOTION', 'STANDARD_PROMO', 'GROWTH_TARGET']
- However, a new class was added in 2023-H1: 'LOW_ENGAGEMENT'
- This class is not defined in LabelEncoder in the .pkl file → so transform() throws an error.
- Harmonizes label encoders across all trained models to ensure compatibility
with unseen labels in newer periods

Academic Purpose:
- Ensure cross-period compatibility for comprehensive model comparison
- Harmonize label encoders to handle evolving labeling strategies
- Maintain academic reproducibility and audit compliance
- Enable fair comparison across all periods without label conflicts

Author: Muhammed Yavuzhan CANLI
Institution: University of Bath
Course: MSc Business Analytics
Academic Standard: A-Grade Compliance
"""

import numpy as np
import pandas as pd
import psycopg2
import logging
import joblib
import glob
import os
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LabelHarmonizer:
    """
    Harmonizes label encoders across all periods to ensure compatibility
    with evolving labeling strategies and unseen labels
    """
    
    def __init__(self, model_dir: str = "models/generic_rf"):
        self.model_dir = model_dir
        self.all_periods = []
        self.all_unique_labels = set()
        self.harmonized_encoder = None
        
        # Academic metadata
        self.academic_metadata = {
            'institution': 'University of Bath',
            'course': 'MSc Business Analytics',
            'project': 'Cross-Period Label Harmonization',
            'academic_standard': 'A-Grade Academic Compliance',
            'methodology': 'Comprehensive label encoder harmonization',
            'harmonization_date': datetime.now().isoformat(),
            'author': 'Muhammed Yavuzhan CANLI'
        }
    
    def discover_all_periods(self):
        """
        Discover all available periods and their model files
        """
        logger.info("Discovering all available periods...")
        
        model_pattern = f"{self.model_dir}/generic_rf_promotion_*_v*.pkl"
        model_files = glob.glob(model_pattern)
        
        period_models = {}
        
        for model_file in model_files:
            try:
                filename = os.path.basename(model_file)
                parts = filename.split('_')
                period = parts[3]  # Extract period like "2022-H1"
                
                # Keep latest version per period
                if period not in period_models:
                    period_models[period] = model_file
                else:
                    # Compare modification times, keep newer
                    if os.path.getmtime(model_file) > os.path.getmtime(period_models[period]):
                        period_models[period] = model_file
                        
            except Exception as e:
                logger.warning(f"Could not parse model file {model_file}: {e}")
                continue
        
        self.all_periods = sorted(period_models.keys())
        logger.info(f"Discovered {len(self.all_periods)} periods: {self.all_periods}")
        
        return period_models
    
    def load_period_data_for_labels(self, period_id: str):
        """
        Load period data specifically to analyze label distribution
        """
        logger.info(f"Loading label data for {period_id}...")
        
        conn = psycopg2.connect(
            host="localhost",
            database="casino_research",
            user="researcher",
            password="academic_password_2024"
        )
        
        query = f"""
        SELECT 
            cf.customer_id,
            cf.total_bet, cf.avg_bet, cf.loss_rate, cf.total_sessions,
            cf.days_since_last_visit, cf.session_duration_volatility,
            cf.loss_chasing_score, cf.sessions_last_30d, cf.bet_trend_ratio,
            ks.cluster_id as kmeans_cluster_id,
            ks.cluster_label as kmeans_segment,
            ks.silhouette_score,
            ks.avg_session_from_metadata as segment_avg_session
        FROM casino_data.customer_features cf
        INNER JOIN casino_data.kmeans_segments ks 
            ON cf.customer_id = ks.customer_id 
            AND cf.analysis_period = ks.period_id
        WHERE cf.analysis_period = '{period_id}'
            AND cf.total_bet > 0
            AND cf.total_bet <= 1500000
            AND ks.kmeans_version = 2
        ORDER BY cf.customer_id
        """
        
        try:
            df = pd.read_sql_query(query, conn)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Failed to load data for period {period_id}: {e}")
            raise
    
    def create_enhanced_features_for_labels(self, df: pd.DataFrame):
        """
        Apply feature engineering to prepare for label creation
        """
        df_enhanced = df.copy()
        
        # Segmentation encoding
        segment_hierarchy = {
            'Casual_Player': 1,
            'Regular_Player': 2, 
            'High_Value_Player': 3,
            'At_Risk_Player': 0
        }
        df_enhanced['kmeans_segment_encoded'] = df_enhanced['kmeans_segment'].map(segment_hierarchy).fillna(1)
        
        # Enhanced features
        df_enhanced['personal_vs_segment_ratio'] = (
            df_enhanced['total_bet'] / df_enhanced['segment_avg_session'].replace(0, 1)
        ).fillna(1.0)
        
        df_enhanced['risk_score'] = (
            df_enhanced['loss_chasing_score'] * 0.3 +
            (df_enhanced['loss_rate'] > 25).astype(int) * 20 +
            (df_enhanced['days_since_last_visit'] > 60).astype(int) * 15 +
            (df_enhanced['personal_vs_segment_ratio'] < 0.5).astype(int) * 10
        )
        
        return df_enhanced
    
    def generate_labels_for_period(self, df: pd.DataFrame):
        """
        Generate labels for a specific period using the same probabilistic approach
        """
        labels = []
        np.random.seed(42)  # Academic reproducibility
        
        # Academic class balance parameters
        min_samples_per_class = max(15, len(df) // 20)
        target_distribution = {
            'NO_PROMOTION': 0.35,
            'STANDARD_PROMO': 0.25,
            'INTERVENTION_NEEDED': 0.15,
            'GROWTH_TARGET': 0.12,
            'HIGH_VALUE_TIER': 0.08,
            'LOW_ENGAGEMENT': 0.05
        }
        
        # Calculate target counts
        total_customers = len(df)
        target_counts = {label: int(total_customers * pct) for label, pct in target_distribution.items()}
        current_counts = {label: 0 for label in target_counts.keys()}
        
        for _, customer in df.iterrows():
            # Probabilistic academic approach
            risk_probability = min(customer['risk_score'] / 100, 0.9)
            intervention_threshold = 0.6
            value_probability = customer['kmeans_segment_encoded'] / 3
            engagement_prob = min(customer['total_sessions'] / 20, 1.0)
            
            # Balanced label assignment
            label_assigned = False
            
            # Priority 1: Risk management
            if (risk_probability > intervention_threshold and 
                current_counts['INTERVENTION_NEEDED'] < target_counts['INTERVENTION_NEEDED']):
                intervention_prob = 0.8 + (risk_probability - intervention_threshold) * 0.5
                if np.random.binomial(1, min(intervention_prob, 0.95)):
                    label = 'INTERVENTION_NEEDED'
                    label_assigned = True
            
            # Priority 2: High value customers
            if (not label_assigned and value_probability > 0.7 and engagement_prob > 0.5 and 
                risk_probability < 0.3 and current_counts['HIGH_VALUE_TIER'] < target_counts['HIGH_VALUE_TIER']):
                high_value_prob = value_probability * engagement_prob * (1 - risk_probability)
                if np.random.binomial(1, min(high_value_prob, 0.8)):
                    label = 'HIGH_VALUE_TIER'
                    label_assigned = True
            
            # Priority 3: Growth targets
            if (not label_assigned and value_probability > 0.4 and 
                customer['personal_vs_segment_ratio'] > 1.1 and
                current_counts['GROWTH_TARGET'] < target_counts['GROWTH_TARGET']):
                growth_prob = value_probability * min(customer['personal_vs_segment_ratio'] / 2, 0.8)
                if np.random.binomial(1, min(growth_prob, 0.7)):
                    label = 'GROWTH_TARGET'
                    label_assigned = True
            
            # Priority 4: Standard promotions
            if (not label_assigned and engagement_prob > 0.15 and risk_probability < 0.5 and
                current_counts['STANDARD_PROMO'] < target_counts['STANDARD_PROMO']):
                standard_prob = engagement_prob * (1 - risk_probability) * 0.7
                if np.random.binomial(1, standard_prob):
                    label = 'STANDARD_PROMO'
                    label_assigned = True
            
            # Priority 5: Low engagement
            if (not label_assigned and (engagement_prob < 0.2 or risk_probability > 0.7) and
                current_counts['LOW_ENGAGEMENT'] < target_counts['LOW_ENGAGEMENT']):
                label = 'LOW_ENGAGEMENT'
                label_assigned = True
            
            # Default: NO_PROMOTION
            if not label_assigned:
                label = 'NO_PROMOTION'
            
            labels.append(label)
            current_counts[label] = current_counts.get(label, 0) + 1
        
        return labels
    
    def discover_all_unique_labels(self):
        """
        Discover all unique labels across all periods
        """
        logger.info("Discovering all unique labels across all periods...")
        
        period_models = self.discover_all_periods()
        all_labels = set()
        
        for period_id in self.all_periods:
            try:
                # Load period data
                df = self.load_period_data_for_labels(period_id)
                df_enhanced = self.create_enhanced_features_for_labels(df)
                
                # Generate labels
                period_labels = self.generate_labels_for_period(df_enhanced)
                period_unique_labels = set(period_labels)
                
                logger.info(f"{period_id}: {len(period_unique_labels)} unique labels: {sorted(period_unique_labels)}")
                all_labels.update(period_unique_labels)
                
            except Exception as e:
                logger.error(f"Failed to analyze labels for {period_id}: {e}")
                continue
        
        self.all_unique_labels = all_labels
        logger.info(f"Total unique labels across all periods: {len(all_labels)}")
        logger.info(f"All unique labels: {sorted(all_labels)}")
        
        return sorted(all_labels)
    
    def create_harmonized_label_encoder(self):
        """
        Create a harmonized label encoder that knows about all possible labels
        """
        logger.info("Creating harmonized label encoder...")
        
        # Discover all unique labels
        all_labels = self.discover_all_unique_labels()
        
        # Create new label encoder with all labels
        self.harmonized_encoder = LabelEncoder()
        self.harmonized_encoder.fit(all_labels)
        
        logger.info(f"Harmonized encoder created with {len(all_labels)} classes:")
        for i, label in enumerate(self.harmonized_encoder.classes_):
            logger.info(f"   {i}: {label}")
        
        return self.harmonized_encoder
    
    def update_model_label_encoders(self, backup: bool = True):
        """
        Update all model files with harmonized label encoder
        """
        logger.info("Updating all model files with harmonized label encoder...")
        
        period_models = self.discover_all_periods()
        
        if not self.harmonized_encoder:
            logger.error("Harmonized encoder not created. Call create_harmonized_label_encoder() first.")
            return False
        
        updated_count = 0
        
        for period_id, model_path in period_models.items():
            try:
                # Create backup if requested
                if backup:
                    backup_path = model_path.replace('.pkl', '_backup_before_harmonization.pkl')
                    import shutil
                    shutil.copy2(model_path, backup_path)
                    logger.info(f"Backup created: {backup_path}")
                
                # Load model package
                model_package = joblib.load(model_path)
                
                # Store original encoder for reference
                original_encoder = model_package['label_encoder']
                original_classes = original_encoder.classes_
                
                # Update with harmonized encoder
                model_package['label_encoder'] = self.harmonized_encoder
                
                # Add harmonization metadata
                model_package['harmonization_metadata'] = {
                    'harmonized_date': datetime.now().isoformat(),
                    'original_classes': original_classes.tolist(),
                    'harmonized_classes': self.harmonized_encoder.classes_.tolist(),
                    'added_classes': list(set(self.harmonized_encoder.classes_) - set(original_classes)),
                    'academic_compliance': 'A-Grade Cross-Period Compatibility'
                }
                
                # Save updated model
                joblib.dump(model_package, model_path)
                
                logger.info(f"Updated {period_id}: Added {len(set(self.harmonized_encoder.classes_) - set(original_classes))} new classes")
                updated_count += 1
                
            except Exception as e:
                logger.error(f"Failed to update {period_id}: {e}")
                continue
        
        logger.info(f"Successfully updated {updated_count} model files with harmonized encoder")
        return updated_count > 0
    
    def generate_harmonization_report(self):
        """
        Generate comprehensive harmonization report
        """
        logger.info("Generating harmonization report...")
        
        period_models = self.discover_all_periods()
        
        print("\nLABEL HARMONIZATION REPORT")
        print("University of Bath - MSc Business Analytics")
        print("Academic Standard: A-Grade Compliance")
        print("=" * 70)
        
        # Report harmonized encoder
        if self.harmonized_encoder:
            print(f"\nHarmonized Label Encoder:")
            print(f"Total Classes: {len(self.harmonized_encoder.classes_)}")
            print(f"Classes: {list(self.harmonized_encoder.classes_)}")
        
        # Check each model's current encoder
        print(f"\nPer-Period Label Encoder Status:")
        print("-" * 50)
        
        for period_id, model_path in period_models.items():
            try:
                model_package = joblib.load(model_path)
                encoder = model_package['label_encoder']
                
                print(f"\n{period_id}:")
                print(f"   Classes: {len(encoder.classes_)}")
                print(f"   Labels: {list(encoder.classes_)}")
                
                if 'harmonization_metadata' in model_package:
                    harm_meta = model_package['harmonization_metadata']
                    print(f"   Status: Harmonized on {harm_meta['harmonized_date'][:10]}")
                    if harm_meta['added_classes']:
                        print(f"   Added: {harm_meta['added_classes']}")
                else:
                    print(f"   Status: Original (not harmonized)")
                
            except Exception as e:
                print(f"\n{period_id}: Error loading - {e}")
        
        # Academic compliance summary
        print(f"\n" + "=" * 50)
        print("ACADEMIC COMPLIANCE SUMMARY")
        print("=" * 50)
        print(f"Institution: University of Bath")
        print(f"Standard: A-Grade Academic Requirements")
        print(f"Methodology: Cross-period label encoder harmonization")
        print(f"Compatibility: All periods now support identical label sets")
        print(f"Reproducibility: Fixed seed and deterministic harmonization")
        
        return True
    
    def save_harmonization_config(self, output_dir: str = "models/harmonization"):
        """
        Save harmonization configuration for audit and reference
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        config_path = f"{output_dir}/label_harmonization_config_{timestamp}.pkl"
        
        harmonization_config = {
            'harmonized_encoder': self.harmonized_encoder,
            'all_periods': self.all_periods,
            'all_unique_labels': sorted(self.all_unique_labels),
            'academic_metadata': self.academic_metadata,
            'harmonization_summary': {
                'total_periods': len(self.all_periods),
                'total_unique_labels': len(self.all_unique_labels),
                'harmonized_classes': self.harmonized_encoder.classes_.tolist() if self.harmonized_encoder else []
            }
        }
        
        joblib.dump(harmonization_config, config_path)
        logger.info(f"Harmonization configuration saved: {config_path}")
        
        return config_path

def main():
    """
    Main execution function for label harmonization
    """
    print("LABEL HARMONIZER FOR ALL PERIODS")
    print("University of Bath - MSc Business Analytics")
    print("Academic Standard: A-Grade Compliance")
    print("=" * 60)
    
    # Initialize harmonizer
    harmonizer = LabelHarmonizer()
    
    try:
        # Create harmonized label encoder
        harmonized_encoder = harmonizer.create_harmonized_label_encoder()
        
        if not harmonized_encoder:
            print("Error: Could not create harmonized encoder")
            return
        
        # Generate initial report
        harmonizer.generate_harmonization_report()
        
        # Ask for confirmation before updating models
        print(f"\nReady to update all model files with harmonized encoder.")
        print(f"This will modify {len(harmonizer.all_periods)} model files.")
        print(f"Backups will be created automatically.")
        
        response = input("\nProceed with harmonization? (y/N): ").lower().strip()
        
        if response == 'y':
            # Update model encoders
            success = harmonizer.update_model_label_encoders(backup=True)
            
            if success:
                print(f"\nHARMONIZATION COMPLETED SUCCESSFULLY!")
                
                # Generate final report
                harmonizer.generate_harmonization_report()
                
                # Save configuration
                config_path = harmonizer.save_harmonization_config()
                
                print(f"\nAcademic Compliance Confirmed:")
                print(f"   All label encoders harmonized across {len(harmonizer.all_periods)} periods")
                print(f"   Cross-period compatibility ensured")
                print(f"   Backup files created for audit trail")
                print(f"   Configuration saved: {config_path}")
                
                print(f"\nNext Step: Run comprehensive model comparison:")
                print(f"   python model_comparison_from_pkls.py")
                
            else:
                print(f"\nHarmonization failed. Check logs for details.")
        else:
            print(f"\nHarmonization cancelled.")
            
    except Exception as e:
        logger.error(f"Harmonization failed: {e}")
        raise

if __name__ == "__main__":
    main()