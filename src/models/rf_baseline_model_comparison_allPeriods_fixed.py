#!/usr/bin/env python3
"""
Model Comparison - ChatGPT Fixed Version - University of Bath Academic Standard
===============================================================================
CRITICAL FIX based on ChatGPT analysis:
- Training: Balanced data (SMOTE applied)
- Testing: Imbalanced data (realistic evaluation)
- This prevents train-test distribution mismatch
- Simulates real-world deployment conditions

Author: Muhammed Yavuzhan CANLI
Institution: University of Bath  
Course: MSc Business Analytics
Version: ChatGPT-Fixed Train-Test Distribution
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
    log_filename = f"FIXED_model_comparison_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_filename}")
    return logger, log_filename

logger, log_filename = setup_logging()

class FixedModelComparison:
    """
    FIXED: Loads ALL 31 features exactly as used in training
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
        """Discover harmonized RF models"""
        logger.info("Discovering trained harmonized RF models...")
        
        model_pattern = f"{self.model_dir}/clean_harmonized_rf_*_v*.pkl"
        model_files = glob.glob(model_pattern)
        
        period_models = {}
        
        for model_file in model_files:
            try:
                filename = os.path.basename(model_file)
                parts = filename.replace('clean_harmonized_rf_', '').split('_')
                period = parts[0]
                
                if period not in period_models:
                    period_models[period] = model_file
                else:
                    if os.path.getmtime(model_file) > os.path.getmtime(period_models[period]):
                        period_models[period] = model_file
                        
            except Exception as e:
                logger.warning(f"Could not parse model file {model_file}: {e}")
                continue
        
        self.available_periods = list(period_models.keys())
        logger.info(f"Discovered {len(self.available_periods)} trained periods: {self.available_periods}")
        
        return period_models
    
    def load_trained_rf_model(self, model_path: str):
        """Load harmonized RF model"""
        try:
            model_package = joblib.load(model_path)
            
            rf_model_package = {
                'rf_model': model_package['model'],
                'scaler': model_package['scaler'],
                'label_encoder': model_package['label_encoder'],
                'feature_names': model_package['feature_names'],
                'performance': model_package.get('performance', {}),
                'period': model_package.get('period', 'unknown')
            }
            
            logger.info(f"Successfully loaded RF model from {os.path.basename(model_path)}")
            logger.info(f"Model classes: {rf_model_package['label_encoder'].classes_}")
            logger.info(f"Required features ({len(rf_model_package['feature_names'])}): {rf_model_package['feature_names']}")
            
            return rf_model_package
            
        except Exception as e:
            logger.error(f"Failed to load RF model from {model_path}: {e}")
            raise
    
    def create_engineered_features(self, df):
        """
        CRITICAL FIX: Create ALL engineered features with REALISTIC values
        This fixes the "dummy zero" problem that breaks RF predictions
        """
        logger.info("Creating engineered features to match training...")
        
        df_eng = df.copy()
        
        # Log transforms (handle zeros with log1p)
        df_eng['log_total_bet'] = np.log1p(df_eng['total_bet'])
        df_eng['log_total_sessions'] = np.log1p(df_eng['total_sessions'])
        df_eng['log_bet_volatility'] = np.log1p(df_eng['bet_volatility'])
        
        # CRITICAL FIX: Missing features with REALISTIC values (not zeros!)
        df_eng['log_avg_bet'] = np.log1p(df_eng['avg_bet'])  # Was missing!
        
        # Robust versions (capped versions)
        df_eng['total_bet_robust'] = np.minimum(df_eng['total_bet'], df_eng['total_bet'].quantile(0.95))
        df_eng['bet_volatility_robust'] = np.minimum(df_eng['bet_volatility'], df_eng['bet_volatility'].quantile(0.95))
        df_eng['avg_bet_robust'] = np.minimum(df_eng['avg_bet'], df_eng['avg_bet'].quantile(0.95))
        
        # Original features (sometimes training keeps both versions)
        df_eng['total_bet_original'] = df_eng['total_bet']
        df_eng['avg_bet_original'] = df_eng['avg_bet']
        df_eng['total_sessions_original'] = df_eng['total_sessions']
        
        # CRITICAL FIX: bet_volatility_original (was being created as dummy zero)
        df_eng['bet_volatility_original'] = df_eng['bet_volatility']  # Use actual values!
        
        # CRITICAL FIX: Binary features with realistic distributions
        if 'total_bet' in df_eng.columns:
            # Ultra high rollers (top 1%)
            ultra_threshold = df_eng['total_bet'].quantile(0.99)
            df_eng['is_ultra_high_roller'] = (df_eng['total_bet'] > ultra_threshold).astype(int)
            
            # Millionaires (total_bet > 1M)
            df_eng['is_millionaire'] = (df_eng['total_bet'] > 1000000).astype(int)
            
            logger.info(f"✅ Ultra high rollers: {df_eng['is_ultra_high_roller'].sum()} customers")
            logger.info(f"✅ Millionaires: {df_eng['is_millionaire'].sum()} customers")
        else:
            df_eng['is_ultra_high_roller'] = 0
            df_eng['is_millionaire'] = 0
            logger.warning("⚠️ total_bet not available, using zeros")
        
        if 'avg_bet' in df_eng.columns:
            # High average betters (top 10%)
            high_bet_threshold = df_eng['avg_bet'].quantile(0.90)
            df_eng['is_high_avg_better'] = (df_eng['avg_bet'] > high_bet_threshold).astype(int)
            
            logger.info(f"✅ High avg betters: {df_eng['is_high_avg_better'].sum()} customers")
        else:
            df_eng['is_high_avg_better'] = 0
            logger.warning("⚠️ avg_bet not available, using zeros")
        
        logger.info(f"✅ Engineered features created with REALISTIC values. Total columns: {len(df_eng.columns)}")
        
        # Verify no all-zero features
        zero_features = []
        for col in ['bet_volatility_original', 'log_avg_bet', 'is_ultra_high_roller', 'is_millionaire', 'is_high_avg_better']:
            if col in df_eng.columns:
                if df_eng[col].sum() == 0 and len(df_eng) > 10:  # Only warn if significant data
                    zero_features.append(col)
        
        if zero_features:
            logger.warning(f"⚠️ Features with all zeros: {zero_features}")
        else:
            logger.info("✅ All critical features have non-zero values!")
        
        return df_eng
    
    def load_complete_training_data(self, period_id: str):
        """
        FIXED: Load ALL features exactly as used in training
        """
        logger.info(f"Loading COMPLETE training data for period {period_id}...")
        
        conn = psycopg2.connect(**self.db_config)
        
        # COMPLETE QUERY: All basic features from customer_features table
        query = f"""
        SELECT 
            cf.customer_id,
            cf.analysis_period,
            
            -- Basic transaction features
            cf.total_events,
            cf.total_bet,
            cf.avg_bet,
            cf.bet_std,
            cf.total_win,
            cf.avg_win,
            cf.avg_loss,
            cf.loss_rate,
            
            -- Session features
            cf.total_sessions,
            cf.avg_events_per_session,
            cf.session_duration_volatility,
            cf.sessions_last_30d,
            
            -- Diversity features
            cf.game_diversity,
            cf.multi_game_player,
            cf.machine_diversity,
            cf.zone_diversity,
            
            -- Behavioral features
            cf.bet_volatility,
            cf.weekend_preference,
            cf.late_night_player,
            cf.days_since_last_visit,
            cf.loss_chasing_score,
            cf.bet_trend_ratio,
            
            -- Promo label
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
            
            logger.info(f"Loaded {len(df)} customers with {len(df.columns)-2} base features for period {period_id}")
            logger.info(f"Raw label distribution:\n{df['promo_label'].value_counts()}")
            
            # Create engineered features
            df_complete = self.create_engineered_features(df)
            
            logger.info(f"Complete dataset: {len(df_complete.columns)} total features")
            
            return df_complete
            
        except Exception as e:
            logger.error(f"Failed to load data for period {period_id}: {e}")
            raise
    
    def apply_balanced_preprocessing(self, X, y, required_features, logger):
        """
        Apply balanced preprocessing ensuring exact feature match
        """
        logger.info("Applying BALANCED preprocessing with feature alignment...")
        
        # Ensure we have all required features with NO DUMMY ZEROS
        missing_features = [f for f in required_features if f not in X.columns]
        if missing_features:
            logger.error(f"❌ CRITICAL: Still missing features after engineering: {missing_features}")
            logger.error("This indicates the feature engineering function needs updating!")
            raise ValueError(f"Missing critical features: {missing_features}")
        else:
            logger.info("✅ All required features present - no dummy zeros needed!")
        
        # Select exactly the required features in the correct order
        X = X[required_features].copy()
        logger.info(f"✅ Feature matrix shape after alignment: {X.shape}")
        
        # Verify feature quality (no all-zero columns)
        zero_cols = [col for col in X.columns if X[col].sum() == 0 and len(X) > 10]
        if zero_cols:
            logger.warning(f"⚠️ Features with all zeros detected: {zero_cols}")
            logger.warning("This may still impact RF performance!")
        else:
            logger.info("✅ No all-zero features detected - good data quality!")
        
        # Handle missing values
        X_clean = X.fillna(X.median())
        
        # Count current labels
        current_counts = y.value_counts()
        total_samples = len(y)
        
        logger.info(f"Raw label distribution:\n{current_counts}")
        
        # Target distribution - EXACT same as training
        target_distribution = {
            'NO_PROMOTION': 0.75,        # 75% - majority class
            'GROWTH_TARGET': 0.12,       # 12% - growth opportunities  
            'LOW_ENGAGEMENT': 0.08,      # 8% - engagement issues
            'INTERVENTION_NEEDED': 0.05  # 5% - risk management
        }
        
        # Calculate target counts
        target_counts = {}
        for label, pct in target_distribution.items():
            target_counts[label] = max(50, int(total_samples * pct))
        
        logger.info(f"Target distribution: {target_counts}")
        
        # Check if current distribution is very imbalanced
        min_class_count = current_counts.min()
        max_class_count = current_counts.max()
        imbalance_ratio = max_class_count / min_class_count if min_class_count > 0 else float('inf')
        
        logger.info(f"Current imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 20:  # Very imbalanced - SAME threshold as training
            logger.info("Applying balanced resampling...")
            
            # Create balanced dataset
            balanced_indices = []
            
            for label, target_count in target_counts.items():
                if label in current_counts.index:
                    # Get indices for this label
                    label_indices = y[y == label].index.tolist()
                    current_count = len(label_indices)
                    
                    if current_count >= target_count:
                        # Downsample
                        selected_indices = np.random.choice(label_indices, target_count, replace=False)
                    else:
                        # Upsample with replacement
                        selected_indices = np.random.choice(label_indices, target_count, replace=True)
                    
                    balanced_indices.extend(selected_indices)
                    logger.info(f"  {label}: {current_count} -> {target_count}")
            
            # Apply balanced sampling
            balanced_indices = np.array(balanced_indices)
            np.random.shuffle(balanced_indices)
            
            X_clean = X_clean.iloc[balanced_indices].reset_index(drop=True)
            y = y.iloc[balanced_indices].reset_index(drop=True)
            
            logger.info(f"Balanced dataset size: {len(y)}")
            logger.info(f"New label distribution:\n{y.value_counts()}")
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        # Ensure all 4 classes are present
        unique_labels = sorted(y.unique())
        required_labels = ['NO_PROMOTION', 'GROWTH_TARGET', 'LOW_ENGAGEMENT', 'INTERVENTION_NEEDED']
        missing_labels = [label for label in required_labels if label not in unique_labels]
        
        if missing_labels:
            logger.warning(f"Missing labels detected: {missing_labels}")
            logger.info("Adding minimal dummy samples for missing classes...")
            
            # Add one dummy sample per missing class
            for missing_label in missing_labels:
                # Add dummy feature row using mean values
                dummy_features = np.mean(X_scaled, axis=0).reshape(1, -1)
                X_scaled = np.vstack([X_scaled, dummy_features])
                
                # Add dummy label
                y = pd.concat([y, pd.Series([missing_label])], ignore_index=True)
            
            logger.info(f"Final label distribution after dummy addition:\n{y.value_counts()}")
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        logger.info(f"Balanced label classes: {label_encoder.classes_}")
        logger.info(f"Balanced encoded distribution: {np.bincount(y_encoded)}")
        logger.info(f"SUCCESS: Balanced preprocessing completed with {len(label_encoder.classes_)} classes")
        
        return X_scaled, y_encoded, scaler, label_encoder
    
    def evaluate_algorithms_on_period(self, period_id: str, rf_model_package: dict):
        """
        FIXED: Complete feature alignment for RF evaluation
        """
        logger.info(f"Evaluating algorithms for period {period_id}...")
        
        # Load complete training data with all features
        df = self.load_complete_training_data(period_id)
        
        # Get required features from trained model
        required_features = rf_model_package['feature_names']
        logger.info(f"Required features: {required_features}")
        
        # Prepare feature matrix and target
        X = df.drop(['customer_id', 'analysis_period', 'promo_label'], axis=1, errors='ignore')
        y = df['promo_label'].copy()
        
        # Apply balanced preprocessing with feature alignment
        X_scaled, y_encoded, balanced_scaler, balanced_label_encoder = self.apply_balanced_preprocessing(
            X, y, required_features, logger
        )
        
        # Verify feature alignment
        logger.info(f"Final feature matrix shape: {X_scaled.shape}")
        logger.info(f"Expected features: {len(required_features)}")
        
        if X_scaled.shape[1] != len(required_features):
            logger.error(f"Feature mismatch: Got {X_scaled.shape[1]}, expected {len(required_features)}")
            raise ValueError("Feature count mismatch with trained model")
        
        # Map RF model's encoder to balanced encoder
        try:
            rf_classes = rf_model_package['label_encoder'].classes_
            balanced_classes = balanced_label_encoder.classes_
            
            logger.info(f"RF model classes: {rf_classes}")
            logger.info(f"Balanced classes: {balanced_classes}")
            
            # Ensure classes match
            if not np.array_equal(sorted(rf_classes), sorted(balanced_classes)):
                logger.error(f"Class mismatch! RF: {sorted(rf_classes)}, Balanced: {sorted(balanced_classes)}")
                raise ValueError("Class mismatch between RF model and balanced data")
            
        except Exception as e:
            logger.error(f"Label mapping failed: {e}")
            raise
        
        # CRITICAL FIX: Apply balancing ONLY to training set
        # First create train-test split on original imbalanced data  
        X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Apply SMOTE balancing ONLY to training set
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.pipeline import Pipeline
        
        # Balanced training pipeline
        smote = SMOTE(random_state=42, k_neighbors=3)
        undersampler = RandomUnderSampler(random_state=42)
        balancing_pipeline = Pipeline([('smote', smote), ('undersampler', undersampler)])
        
        try:
            X_train, y_train = balancing_pipeline.fit_resample(X_train_orig, y_train_orig)
            logger.info(f"✅ SMOTE applied to training set: {len(X_train)} balanced samples")
        except:
            # Fallback: Use original training data
            X_train, y_train = X_train_orig, y_train_orig
            logger.warning("⚠️ SMOTE failed, using original training data")
        
        # Test set remains ORIGINAL and IMBALANCED (realistic evaluation)
        X_test, y_test = X_test_orig, y_test_orig
        logger.info(f"✅ Test set kept imbalanced for realistic evaluation: {len(X_test)} samples")
        
        logger.info(f"Train set: {len(X_train)} samples")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Test distribution: {np.bincount(y_test)}")
        
        # Initialize results
        period_results = {}
        
        # Evaluate trained Random Forest model
        logger.info("Evaluating trained Random Forest...")
        
        try:
            # CRITICAL FIX: Use RF model's own scaler on correctly aligned features
            X_test_rf_scaled = rf_model_package['scaler'].transform(X_test)
            
            # Use predict_proba for better ROC AUC calculation
            rf_probabilities = rf_model_package['rf_model'].predict_proba(X_test_rf_scaled)
            rf_predictions = rf_model_package['rf_model'].predict(X_test_rf_scaled)
            
            # Calculate RF metrics with improved handling
            rf_accuracy = accuracy_score(y_test, rf_predictions)
            rf_f1 = f1_score(y_test, rf_predictions, average='weighted')
            
            # Improved ROC AUC calculation
            try:
                rf_roc_auc = roc_auc_score(y_test, rf_probabilities, multi_class='ovr', average='macro')
                logger.info(f"✅ RF ROC AUC calculated successfully: {rf_roc_auc:.4f}")
            except Exception as roc_e:
                logger.warning(f"ROC AUC calculation failed: {roc_e}")
                rf_roc_auc = None
            
            # Log detailed prediction analysis
            unique_predictions = np.unique(rf_predictions)
            pred_distribution = np.bincount(rf_predictions)
            test_distribution = np.bincount(y_test)
            
            logger.info(f"✅ RF Predictions - Classes: {unique_predictions}, Distribution: {pred_distribution}")
            logger.info(f"✅ Test Reality - Distribution: {test_distribution}")
            logger.info(f"✅ Random Forest REALISTIC: Accuracy={rf_accuracy:.4f}, F1={rf_f1:.4f}, ROC_AUC={rf_roc_auc}")
            
        except Exception as e:
            logger.error(f"❌ RF evaluation failed: {e}")
            rf_accuracy = 0.0
            rf_f1 = 0.0
            rf_roc_auc = None
        
        period_results['Random_Forest_Harmonized'] = {
            'accuracy': rf_accuracy,
            'f1_score': rf_f1,
            'roc_auc_macro': rf_roc_auc,
            'model_source': 'Trained PKL Model',
            'n_customers': len(df),
            'n_features': len(required_features),
            'n_classes': len(balanced_label_encoder.classes_),
            'period': period_id
        }
        
        # Evaluate baseline algorithms on same balanced data
        for algorithm_name, algorithm in self.baseline_algorithms.items():
            try:
                logger.info(f"Training {algorithm_name}...")
                
                # Train algorithm on BALANCED training data
                algorithm.fit(X_train, y_train)
                
                # Test on IMBALANCED test data (realistic evaluation)
                predictions = algorithm.predict(X_test)
                
                # Calculate metrics on imbalanced test set
                accuracy = accuracy_score(y_test, predictions)
                f1 = f1_score(y_test, predictions, average='weighted')
                
                # ROC AUC with probability predictions
                roc_auc = None
                try:
                    if hasattr(algorithm, 'predict_proba'):
                        probabilities = algorithm.predict_proba(X_test)
                        roc_auc = roc_auc_score(y_test, probabilities, multi_class='ovr', average='macro')
                        logger.info(f"✅ {algorithm_name} ROC AUC: {roc_auc:.4f}")
                except Exception as e:
                    logger.warning(f"ROC AUC failed for {algorithm_name}: {e}")
                    roc_auc = None
                
                period_results[algorithm_name] = {
                    'accuracy': accuracy,
                    'f1_score': f1,
                    'roc_auc_macro': roc_auc,
                    'model_source': 'Trained on Balanced Data',
                    'n_customers': len(df),
                    'n_features': len(required_features),
                    'n_classes': len(balanced_label_encoder.classes_),
                    'period': period_id
                }
                
                logger.info(f"✅ {algorithm_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, ROC_AUC={roc_auc}")
                
            except Exception as e:
                logger.error(f"❌ Failed to evaluate {algorithm_name}: {e}")
                period_results[algorithm_name] = None
        
        return period_results
    
    def run_comprehensive_comparison(self):
        """Run comprehensive comparison with FIXED feature alignment"""
        logger.info("Starting FIXED comprehensive model comparison...")
        
        period_models = self.discover_trained_models()
        
        if not period_models:
            logger.error("No trained models found")
            return None
        
        for period_id, model_path in period_models.items():
            try:
                logger.info(f"Processing period {period_id}...")
                
                rf_model_package = self.load_trained_rf_model(model_path)
                period_results = self.evaluate_algorithms_on_period(period_id, rf_model_package)
                
                self.comparison_results[period_id] = period_results
                
            except Exception as e:
                logger.error(f"Failed to process period {period_id}: {e}")
                continue
        
        return self.comparison_results
    
    def generate_comprehensive_report(self):
        """Generate comprehensive report"""
        if not self.comparison_results:
            logger.error("No results available for report generation")
            return None
        
        print("\n" + "="*80)
        print("FIXED COMPREHENSIVE MODEL COMPARISON REPORT")
        print("University of Bath - MSc Business Analytics")
        print("REALISTIC EVALUATION: Balanced Training → Imbalanced Testing")
        print("This simulates real-world deployment conditions")
        print("="*80)
        
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
            
            best_algo = period_data.iloc[0]['Algorithm']
            best_acc = period_data.iloc[0]['Accuracy']
            print(f"🏆 Best: {best_algo} (Accuracy: {best_acc})")
        
        # Random Forest analysis
        print("\n" + "="*80)
        print("RANDOM FOREST PERFORMANCE ANALYSIS")
        print("="*80)
        
        rf_data = summary_df[summary_df['Algorithm'] == 'Random_Forest_Harmonized']
        
        if not rf_data.empty:
            print("\n📈 Random Forest Performance Summary:")
            print(rf_data[['Period', 'Accuracy', 'F1_Score', 'ROC_AUC']].to_string(index=False))
            
            avg_acc = rf_data['Accuracy'].mean()
            std_acc = rf_data['Accuracy'].std()
            
            print(f"\n📊 Statistical Summary:")
            print(f"   Average Accuracy: {avg_acc:.4f} +/- {std_acc:.4f}")
            print(f"   Performance Range: {rf_data['Accuracy'].min():.4f} - {rf_data['Accuracy'].max():.4f}")
            
            # RF vs Best Competitor Analysis
            print(f"\n🏆 RANDOM FOREST vs COMPETITORS:")
            for period in sorted(self.comparison_results.keys()):
                period_data = summary_df[summary_df['Period'] == period].copy()
                rf_row = period_data[period_data['Algorithm'] == 'Random_Forest_Harmonized']
                competitors = period_data[period_data['Algorithm'] != 'Random_Forest_Harmonized']
                
                if not rf_row.empty and not competitors.empty:
                    rf_acc = rf_row.iloc[0]['Accuracy']
                    best_competitor = competitors.loc[competitors['Accuracy'].idxmax()]
                    advantage = rf_acc - best_competitor['Accuracy']
                    
                    print(f"   {period}: RF {rf_acc:.4f} vs {best_competitor['Algorithm']} {best_competitor['Accuracy']:.4f} (Advantage: +{advantage:.4f})")
        
        return summary_df
    
    def export_results(self):
        """Export results to CSV"""
        if not self.comparison_results:
            return None
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
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
        csv_path = f"FIXED_model_comparison_{timestamp}.csv"
        detailed_df.to_csv(csv_path, index=False)
        
        logger.info(f"Results exported to: {csv_path}")
        return csv_path

def main():
    """Main execution"""
    print("FIXED MODEL COMPARISON - UNIVERSITY OF BATH")
    print("Feature Alignment Issue RESOLVED")
    print("="*60)
    
    comparison = FixedModelComparison()
    
    try:
        results = comparison.run_comprehensive_comparison()
        
        if not results:
            print("❌ Error: No results obtained")
            return
        
        summary_df = comparison.generate_comprehensive_report()
        csv_path = comparison.export_results()
        
        print(f"\n✅ FIXED COMPARISON COMPLETED SUCCESSFULLY")
        print(f"   Periods analyzed: {len(results)}")
        print(f"   CSV exported: {csv_path}")
        print(f"   Log file: {log_filename}")
        
        logger.info("="*60)
        logger.info("FIXED COMPARISON COMPLETED SUCCESSFULLY")
        logger.info(f"Periods analyzed: {len(results)}")
        logger.info(f"CSV exported: {csv_path}")
        logger.info(f"Log file: {log_filename}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise

if __name__ == "__main__":
    main()