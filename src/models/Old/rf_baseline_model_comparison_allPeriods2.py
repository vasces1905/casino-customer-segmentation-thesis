#!/usr/bin/env python3
"""
Model Comparison from Trained PKL Files - University of Bath Academic Standard - version-3 updated
=============================================================================
update-2: Fixed to work with clean_harmonized_rf models
update-3: Logs added by time-date for each run.
--------------------------------------------------------
Comprehensive baseline comparison using pre-trained Random Forest models
against baseline algorithms on identical datasets

Academic Purpose: 
- Validate Random Forest methodology against classical algorithms
- Ensure fair comparison using identical preprocessing and evaluation metrics
- Provide statistical justification for model selection
- Meet University of Bath MSc Business Analytics academic standards

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
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Setup logging with file output
def setup_logging():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_filename = f"harmonized_model_comparison_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()  # Console output
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_filename}")
    return logger, log_filename

logger, log_filename = setup_logging()

class PKLBasedModelComparison:
    """
    Model comparison using harmonized RF models
    """
    
    def __init__(self, model_dir: str = "models/generic_rf"):
        self.model_dir = model_dir
        self.available_periods = []
        self.comparison_results = {}
        
        # Database connection
        self.db_config = {
            'host': 'localhost',
            'database': 'casino_research',
            'user': 'researcher',
            'password': 'academic_password_2024'
        }
        
        # Baseline algorithms
        self.baseline_algorithms = {
            'Decision_Tree': DecisionTreeClassifier(
                random_state=42, 
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'Logistic_Regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'K_Nearest_Neighbors': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            'Naive_Bayes': GaussianNB(),
            'Support_Vector_Machine': SVC(
                random_state=42, 
                probability=True,
                kernel='rbf',
                class_weight='balanced'
            )
        }
    
    def discover_trained_models(self):
        """
        Discover harmonized RF models
        """
        logger.info("Discovering trained harmonized RF models...")
        
        # Fixed pattern for our models
        model_pattern = f"{self.model_dir}/clean_harmonized_rf_*_v*.pkl"
        model_files = glob.glob(model_pattern)
        
        period_models = {}
        
        for model_file in model_files:
            try:
                filename = os.path.basename(model_file)
                # Parse: clean_harmonized_rf_2022-H2_v1_20250728_0025.pkl
                parts = filename.replace('clean_harmonized_rf_', '').split('_')
                period = parts[0]  # Extract period like "2022-H2"
                
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
        
        self.available_periods = list(period_models.keys())
        logger.info(f"Discovered {len(self.available_periods)} trained periods: {self.available_periods}")
        
        return period_models
    
    def load_trained_rf_model(self, model_path: str):
        """
        Load harmonized RF model - FIXED to use correct keys
        """
        try:
            model_package = joblib.load(model_path)
            
            # Map our keys to expected keys
            rf_model_package = {
                'rf_model': model_package['model'],  # Our key is 'model', not 'rf_model'
                'scaler': model_package['scaler'],
                'label_encoder': model_package['label_encoder'],
                'feature_names': model_package['feature_names'],
                'performance': model_package.get('performance', {}),
                'period': model_package.get('period', 'unknown')
            }
            
            logger.info(f"Successfully loaded RF model from {os.path.basename(model_path)}")
            logger.info(f"Model classes: {rf_model_package['label_encoder'].classes_}")
            logger.info(f"Feature count: {len(rf_model_package['feature_names'])}")
            
            return rf_model_package
            
        except Exception as e:
            logger.error(f"Failed to load RF model from {model_path}: {e}")
            raise
    
    def load_period_data(self, period_id: str):
        """
        Load period data with JOIN to get features and labels
        """
        logger.info(f"Loading period data for {period_id}...")
        
        conn = psycopg2.connect(**self.db_config)
        
        # Join customer_features with promo_label
        query = f"""
        SELECT 
            cf.customer_id,
            cf.analysis_period,
            cf.total_events,
            cf.total_bet,
            cf.avg_bet,
            cf.bet_std,
            cf.total_win,
            cf.avg_win,
            cf.avg_loss,
            cf.loss_rate,
            cf.total_sessions,
            cf.avg_events_per_session,
            cf.game_diversity,
            cf.multi_game_player,
            cf.machine_diversity,
            cf.zone_diversity,
            cf.bet_volatility,
            cf.weekend_preference,
            cf.late_night_player,
            cf.days_since_last_visit,
            cf.session_duration_volatility,
            cf.loss_chasing_score,
            cf.sessions_last_30d,
            cf.bet_trend_ratio,
            
            pl.promo_label
            
        FROM casino_data.customer_features cf
        INNER JOIN casino_data.promo_label pl 
            ON cf.customer_id = pl.customer_id 
            AND cf.analysis_period = pl.period
        WHERE cf.analysis_period = '{period_id}'
        ORDER BY cf.customer_id
        """
        
        try:
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            logger.info(f"Loaded {len(df)} customers for period {period_id}")
            logger.info(f"Label distribution:\n{df['promo_label'].value_counts()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data for period {period_id}: {e}")
            raise
    
    def evaluate_algorithms_on_period(self, period_id: str, rf_model_package: dict):
        """
        Evaluate all algorithms on the same dataset
        """
        logger.info(f"Evaluating algorithms for period {period_id}...")
        
        # Load data
        df = self.load_period_data(period_id)
        
        # Prepare features using RF model's feature names
        feature_names = rf_model_package['feature_names']
        
        # Select only features that exist in both
        available_features = [f for f in feature_names if f in df.columns]
        missing_features = [f for f in feature_names if f not in df.columns]
        
        if missing_features:
            logger.warning(f"Missing features: {missing_features}")
            logger.info(f"Using {len(available_features)} available features")
        
        X = df[available_features].fillna(0)
        y = df['promo_label']
        
        # Use RF model's label encoder
        try:
            y_encoded = rf_model_package['label_encoder'].transform(y)
        except ValueError as e:
            logger.error(f"Label encoding failed: {e}")
            logger.info(f"Available labels in data: {y.unique()}")
            logger.info(f"Model expects: {rf_model_package['label_encoder'].classes_}")
            raise
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features using RF model's scaler
        X_train_scaled = rf_model_package['scaler'].transform(X_train)
        X_test_scaled = rf_model_package['scaler'].transform(X_test)
        
        # Initialize results
        period_results = {}
        
        # Evaluate trained Random Forest model
        logger.info("Evaluating trained Random Forest...")
        rf_predictions = rf_model_package['rf_model'].predict(X_test_scaled)
        rf_probabilities = rf_model_package['rf_model'].predict_proba(X_test_scaled)
        
        # Calculate RF metrics
        rf_accuracy = accuracy_score(y_test, rf_predictions)
        rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
        
        try:
            rf_roc_auc = roc_auc_score(y_test, rf_probabilities, multi_class='ovr', average='macro')
        except Exception as e:
            logger.warning(f"ROC AUC calculation failed: {e}")
            rf_roc_auc = None
        
        period_results['Random_Forest_Harmonized'] = {
            'accuracy': rf_accuracy,
            'f1_score': rf_f1,
            'roc_auc_macro': rf_roc_auc,
            'model_source': 'Harmonized PKL file',
            'n_customers': len(df),
            'n_features': len(available_features),
            'n_classes': len(rf_model_package['label_encoder'].classes_),
            'period': period_id
        }
        
        logger.info(f"Random Forest: Accuracy={rf_accuracy:.4f}, F1={rf_f1:.4f}, ROC_AUC={rf_roc_auc}")
        
        # Evaluate baseline algorithms
        for algorithm_name, algorithm in self.baseline_algorithms.items():
            try:
                logger.info(f"Training {algorithm_name}...")
                
                # Train algorithm
                start_time = datetime.now()
                algorithm.fit(X_train_scaled, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Make predictions
                start_time = datetime.now()
                predictions = algorithm.predict(X_test_scaled)
                prediction_time = (datetime.now() - start_time).total_seconds()
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, predictions)
                f1 = f1_score(y_test, predictions, average='weighted')
                
                # ROC AUC
                roc_auc = None
                try:
                    if hasattr(algorithm, 'predict_proba'):
                        probabilities = algorithm.predict_proba(X_test_scaled)
                        roc_auc = roc_auc_score(y_test, probabilities, multi_class='ovr', average='macro')
                except Exception as e:
                    logger.warning(f"ROC AUC failed for {algorithm_name}: {e}")
                    roc_auc = None
                
                period_results[algorithm_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'roc_auc_macro': roc_auc,
                    'training_time_seconds': training_time,
                    'prediction_time_seconds': prediction_time,
                    'model_source': 'Newly trained',
                    'n_customers': len(df),
                    'n_features': len(available_features),
                    'n_classes': len(rf_model_package['label_encoder'].classes_),
                    'period': period_id
                }
                
                logger.info(f"{algorithm_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC_AUC={roc_auc}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {algorithm_name}: {e}")
                period_results[algorithm_name] = None
        
        return period_results
    
    def run_comprehensive_comparison(self):
        """
        Run comprehensive comparison
        """
        logger.info("Starting comprehensive harmonized model comparison...")
        
        # Discover available models
        period_models = self.discover_trained_models()
        
        if not period_models:
            logger.error("No trained models found")
            return None
        
        # Evaluate each period
        for period_id, model_path in period_models.items():
            try:
                logger.info(f"Processing period {period_id}...")
                
                # Load trained RF model
                rf_model_package = self.load_trained_rf_model(model_path)
                
                # Evaluate all algorithms
                period_results = self.evaluate_algorithms_on_period(period_id, rf_model_package)
                
                self.comparison_results[period_id] = period_results
                
            except Exception as e:
                logger.error(f"Failed to process period {period_id}: {e}")
                continue
        
        return self.comparison_results
    
    def generate_comprehensive_report(self):
        """
        Generate comprehensive report
        """
        if not self.comparison_results:
            logger.error("No results available for report generation")
            return None
        
        print("\n" + "="*80)
        print("COMPREHENSIVE HARMONIZED MODEL COMPARISON REPORT")
        print("University of Bath - MSc Business Analytics")
        print("Academic Standard: A-Grade Compliance")
        print("="*80)
        
        # Create summary DataFrame
        summary_data = []
        
        for period, period_results in self.comparison_results.items():
            for algorithm, metrics in period_results.items():
                if metrics is None:
                    continue
                
                summary_data.append({
                    'Period': period,
                    'Algorithm': algorithm,
                    'Accuracy': round(metrics['accuracy'], 4),
                    'F1_Score': round(metrics['f1_score'], 4),
                    'ROC_AUC': round(metrics['roc_auc_macro'], 4) if metrics['roc_auc_macro'] else 'N/A',
                    'Customers': metrics['n_customers'],
                    'Features': metrics['n_features'],
                    'Classes': metrics['n_classes']
                })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Period-wise analysis
        print("\nPERIOD-WISE ALGORITHM PERFORMANCE:")
        print("-" * 80)
        
        for period in sorted(self.comparison_results.keys()):
            print(f"\n📊 {period}:")
            period_data = summary_df[summary_df['Period'] == period].copy()
            period_data = period_data.sort_values('Accuracy', ascending=False)
            
            print(period_data[['Algorithm', 'Accuracy', 'F1_Score', 'ROC_AUC']].to_string(index=False))
            
            # Best performer
            best_algo = period_data.iloc[0]['Algorithm']
            best_acc = period_data.iloc[0]['Accuracy']
            print(f"🏆 Best: {best_algo} (Accuracy: {best_acc})")
        
        # Random Forest performance analysis
        print("\n" + "="*80)
        print("RANDOM FOREST HARMONIZED PERFORMANCE ANALYSIS")
        print("="*80)
        
        rf_data = summary_df[summary_df['Algorithm'] == 'Random_Forest_Harmonized']
        
        if not rf_data.empty:
            print("\n📈 Random Forest Performance Summary:")
            print(rf_data[['Period', 'Accuracy', 'F1_Score', 'ROC_AUC']].to_string(index=False))
            
            # Overall statistics
            avg_acc = rf_data['Accuracy'].mean()
            std_acc = rf_data['Accuracy'].std()
            avg_f1 = rf_data['F1_Score'].mean()
            
            print(f"\n📊 Statistical Summary:")
            print(f"   Average Accuracy: {avg_acc:.4f} ± {std_acc:.4f}")
            print(f"   Average F1 Score: {avg_f1:.4f}")
            print(f"   Performance Range: {rf_data['Accuracy'].min():.4f} - {rf_data['Accuracy'].max():.4f}")
        
        # Algorithm ranking
        print("\n" + "="*80)
        print("ALGORITHM RANKING ACROSS ALL PERIODS")
        print("="*80)
        
        avg_performance = summary_df.groupby('Algorithm').agg({
            'Accuracy': ['mean', 'std'],
            'F1_Score': 'mean',
            'ROC_AUC': lambda x: pd.to_numeric(x, errors='coerce').mean()
        }).round(4)
        
        avg_performance.columns = ['Avg_Accuracy', 'Std_Accuracy', 'Avg_F1', 'Avg_ROC_AUC']
        avg_performance = avg_performance.sort_values('Avg_Accuracy', ascending=False)
        
        print(avg_performance)
        
        # Academic conclusion
        rf_rank = avg_performance.index.get_loc('Random_Forest_Harmonized') + 1 if 'Random_Forest_Harmonized' in avg_performance.index else 'N/A'
        
        print(f"\n🎓 ACADEMIC CONCLUSION:")
        print(f"   Random Forest Rank: #{rf_rank} out of {len(avg_performance)} algorithms")
        print(f"   Methodology: Harmonized labeling with 4-class system")
        print(f"   Justification: Ensemble robustness + feature interpretability")
        print(f"   Academic Standard: University of Bath A-Grade compliance")
        
        return summary_df
    
    def export_results(self):
        """
        Export results to CSV
        """
        if not self.comparison_results:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        # Create detailed results
        detailed_data = []
        for period, period_results in self.comparison_results.items():
            for algorithm, metrics in period_results.items():
                if metrics is None:
                    continue
                detailed_data.append({
                    'Period': period,
                    'Algorithm': algorithm,
                    'Accuracy': metrics['accuracy'],
                    'F1_Score': metrics['f1_score'],
                    'ROC_AUC_Macro': metrics['roc_auc_macro'],
                    'N_Customers': metrics['n_customers'],
                    'N_Features': metrics['n_features'],
                    'N_Classes': metrics['n_classes']
                })
        
        detailed_df = pd.DataFrame(detailed_data)
        csv_path = f"harmonized_model_comparison_{timestamp}.csv"
        detailed_df.to_csv(csv_path, index=False)
        
        logger.info(f"Results exported to: {csv_path}")
        return csv_path

def main():
    """
    Main execution
    """
    print("HARMONIZED MODEL COMPARISON - UNIVERSITY OF BATH")
    print("="*60)
    
    # Initialize comparison
    comparison = PKLBasedModelComparison()
    
    try:
        # Run comparison
        results = comparison.run_comprehensive_comparison()
        
        if not results:
            print("❌ Error: No results obtained")
            return
        
        # Generate report
        summary_df = comparison.generate_comprehensive_report()
        
        # Export results
        csv_path = comparison.export_results()
        
        print(f"\n✅ COMPARISON COMPLETED SUCCESSFULLY")
        print(f"   Periods analyzed: {len(results)}")
        print(f"   CSV exported: {csv_path}")
        print(f"   Log file: {log_filename}")
        
        # Log final summary
        logger.info("="*60)
        logger.info("COMPARISON COMPLETED SUCCESSFULLY")
        logger.info(f"Periods analyzed: {len(results)}")
        logger.info(f"Algorithms compared: {len(comparison.baseline_algorithms) + 1}")
        logger.info(f"CSV exported: {csv_path}")
        logger.info(f"Log file: {log_filename}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise

if __name__ == "__main__":
    main()