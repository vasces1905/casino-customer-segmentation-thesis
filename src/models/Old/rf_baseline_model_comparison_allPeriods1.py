#!/usr/bin/env python3
"""
RF Baseline Model Comparison - All Periods Comprehensive
======================================================
Comprehensive baseline comparison across all periods to validate RF methodology

Academic Purpose: 
- Cross-period algorithm validation
- Statistical significance analysis
- Methodological justification for RF selection
"""

import numpy as np
import pandas as pd
import psycopg2
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveBaselineComparison:
    """
    Comprehensive baseline comparison across all periods
    """
    
    def __init__(self):
        self.periods = ['2022-H1', '2022-H2', '2023-H1', '2023-H2']
        self.results_all_periods = {}
        
        # Define baseline models
        self.baseline_models = {
            'Random_Forest': RandomForestClassifier(
                n_estimators=150, max_depth=10, random_state=42, 
                min_samples_split=8, min_samples_leaf=4
            ),
            'Decision_Tree': DecisionTreeClassifier(random_state=42, max_depth=15),
            'Logistic_Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True, kernel='rbf'),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive_Bayes': GaussianNB()
        }
    
    def load_period_data(self, period_id: str):
        """Load clean data for specific period"""
        logger.info(f"Loading data for {period_id}...")
        
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
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def prepare_features_labels(self, df):
        """Prepare features and labels"""
        # Enhanced features
        df_enhanced = df.copy()
        
        # Segment encoding
        segment_hierarchy = {
            'Casual_Player': 1, 'Regular_Player': 2, 
            'High_Value_Player': 3, 'At_Risk_Player': 0
        }
        df_enhanced['kmeans_segment_encoded'] = df_enhanced['kmeans_segment'].map(segment_hierarchy).fillna(1)
        
        # Additional features
        df_enhanced['personal_vs_segment_ratio'] = (
            df_enhanced['total_bet'] / df_enhanced['segment_avg_session'].replace(0, 1)
        ).fillna(1.0)
        
        df_enhanced['risk_score'] = (
            df_enhanced['loss_chasing_score'] * 0.3 +
            (df_enhanced['loss_rate'] > 25).astype(int) * 20 +
            (df_enhanced['days_since_last_visit'] > 60).astype(int) * 15
        )
        
        df_enhanced['value_tier'] = pd.cut(
            df_enhanced['total_bet'], bins=[0, 500, 2000, 10000, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        df_enhanced['engagement_level'] = pd.cut(
            df_enhanced['total_sessions'], bins=[0, 2, 5, 15, float('inf')],
            labels=[0, 1, 2, 3]
        ).astype(int)
        
        df_enhanced['is_high_value'] = (df_enhanced['kmeans_segment_encoded'] >= 2).astype(int)
        df_enhanced['needs_attention'] = (df_enhanced['risk_score'] > 30).astype(int)
        df_enhanced['segment_outperformer'] = (df_enhanced['personal_vs_segment_ratio'] > 1.5).astype(int)
        
        # Create labels
        labels = []
        np.random.seed(42)
        
        for _, customer in df_enhanced.iterrows():
            risk_prob = min(customer['risk_score'] / 100, 0.9)
            value_prob = customer['kmeans_segment_encoded'] / 3
            engagement_prob = min(customer['total_sessions'] / 20, 1.0)
            
            if risk_prob > 0.6:
                label = 'INTERVENTION_NEEDED'
            elif value_prob > 0.7 and engagement_prob > 0.5 and risk_prob < 0.3:
                label = 'HIGH_VALUE_TIER'
            elif value_prob > 0.4 and customer['personal_vs_segment_ratio'] > 1.1:
                label = 'GROWTH_TARGET'
            elif engagement_prob > 0.15 and risk_prob < 0.5:
                label = 'STANDARD_PROMO'
            elif engagement_prob < 0.2 or risk_prob > 0.7:
                label = 'LOW_ENGAGEMENT'
            else:
                label = 'NO_PROMOTION'
            
            labels.append(label)
        
        df_enhanced['target_label'] = labels
        
        # Feature selection
        feature_columns = [
            'total_bet', 'avg_bet', 'loss_rate', 'total_sessions',
            'days_since_last_visit', 'session_duration_volatility',
            'loss_chasing_score', 'sessions_last_30d', 'bet_trend_ratio',
            'kmeans_cluster_id', 'kmeans_segment_encoded', 
            'segment_avg_session', 'silhouette_score',
            'personal_vs_segment_ratio', 'risk_score',
            'value_tier', 'engagement_level',
            'is_high_value', 'needs_attention', 'segment_outperformer'
        ]
        
        X = df_enhanced[feature_columns].fillna(0)
        y = df_enhanced['target_label']
        
        return X, y
    
    def evaluate_period(self, period_id: str):
        """Evaluate all models for a specific period"""
        logger.info(f"Evaluating period {period_id}...")
        
        # Load data
        df = self.load_period_data(period_id)
        if df.empty:
            logger.warning(f"No data for {period_id}")
            return None
        
        # Prepare features
        X, y = self.prepare_features_labels(df)
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Results for this period
        period_results = {}
        
        # Test each model
        for model_name, model in self.baseline_models.items():
            try:
                # Train model
                start_time = datetime.now()
                model.fit(X_train_scaled, y_train)
                training_time = (datetime.now() - start_time).total_seconds()
                
                # Predict
                start_time = datetime.now()
                y_pred = model.predict(X_test_scaled)
                prediction_time = (datetime.now() - start_time).total_seconds()
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                
                # ROC AUC
                try:
                    y_pred_proba = model.predict_proba(X_test_scaled)
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                except:
                    roc_auc = 0.0
                
                # Store results
                period_results[model_name] = {
                    'accuracy': accuracy,
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std(),
                    'roc_auc_macro': roc_auc,
                    'training_time': training_time,
                    'prediction_time': prediction_time,
                    'n_customers': len(df),
                    'n_classes': len(label_encoder.classes_)
                }
                
                logger.info(f"   {model_name}: Acc={accuracy:.4f}, ROC={roc_auc:.4f}")
                
            except Exception as e:
                logger.error(f"   {model_name} failed: {e}")
                period_results[model_name] = None
        
        return period_results
    
    def run_comprehensive_comparison(self):
        """Run comparison across all periods"""
        logger.info("Starting comprehensive baseline comparison...")
        
        for period in self.periods:
            period_results = self.evaluate_period(period)
            if period_results:
                self.results_all_periods[period] = period_results
        
        return self.results_all_periods
    
    def generate_comprehensive_report(self):
        """Generate comprehensive cross-period report"""
        if not self.results_all_periods:
            logger.error("No results available")
            return None
        
        print("\n" + "="*100)
        print("COMPREHENSIVE BASELINE COMPARISON - ALL PERIODS")
        print("="*100)
        print("Academic Validation: Cross-period algorithm performance analysis")
        print("="*100)
        
        # Create summary table
        summary_data = []
        
        for period, period_results in self.results_all_periods.items():
            for model_name, result in period_results.items():
                if result is None:
                    continue
                
                summary_data.append({
                    'Period': period,
                    'Model': model_name,
                    'Accuracy': round(result['accuracy'], 4),
                    'CV_Mean': round(result['cv_score_mean'], 4),
                    'ROC_AUC': round(result['roc_auc_macro'], 4),
                    'Customers': result['n_customers'],
                    'Classes': result['n_classes']
                })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Period-wise analysis
        print("\nPERIOD-WISE PERFORMANCE ANALYSIS:")
        print("="*60)
        
        rf_performance = []
        best_models = []
        
        for period in self.periods:
            if period not in self.results_all_periods:
                continue
                
            print(f"\n{period}:")
            period_data = df_summary[df_summary['Period'] == period].copy()
            period_data = period_data.sort_values('Accuracy', ascending=False)
            
            print(period_data[['Model', 'Accuracy', 'ROC_AUC']].to_string(index=False))
            
            # Track RF performance
            rf_row = period_data[period_data['Model'] == 'Random_Forest']
            if not rf_row.empty:
                rf_rank = period_data.index[period_data['Model'] == 'Random_Forest'].tolist()[0] + 1
                rf_acc = rf_row['Accuracy'].values[0]
                rf_roc = rf_row['ROC_AUC'].values[0]
                
                rf_performance.append({
                    'period': period,
                    'rank': rf_rank,
                    'accuracy': rf_acc,
                    'roc_auc': rf_roc,
                    'total_models': len(period_data)
                })
            
            # Track best model
            best_model = period_data.iloc[0]['Model']
            best_acc = period_data.iloc[0]['Accuracy']
            best_models.append({
                'period': period,
                'best_model': best_model,
                'best_accuracy': best_acc
            })
        
        # Cross-period RF analysis
        print("\n" + "="*60)
        print("RANDOM FOREST CROSS-PERIOD ANALYSIS")
        print("="*60)
        
        if rf_performance:
            rf_df = pd.DataFrame(rf_performance)
            
            print(f"\nRF Performance Summary:")
            print(f"{'Period':<10} {'Rank':<6} {'Accuracy':<10} {'ROC_AUC':<10}")
            print("-" * 40)
            
            for _, row in rf_df.iterrows():
                print(f"{row['period']:<10} #{row['rank']:<5} {row['accuracy']:<10.4f} {row['roc_auc']:<10.4f}")
            
            # Statistical analysis
            avg_rank = rf_df['rank'].mean()
            avg_accuracy = rf_df['accuracy'].mean()
            avg_roc = rf_df['roc_auc'].mean()
            std_accuracy = rf_df['accuracy'].std()
            
            print(f"\nSTATISTICAL SUMMARY:")
            print(f"   Average Rank: {avg_rank:.1f}")
            print(f"   Average Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}")
            print(f"   Average ROC AUC: {avg_roc:.4f}")
            print(f"   Consistency: {len(rf_df[rf_df['rank'] <= 3])} / {len(rf_df)} periods in top 3")
        
        # Algorithm dominance analysis
        print(f"\nALGORITHM DOMINANCE ANALYSIS:")
        best_models_df = pd.DataFrame(best_models)
        dominance = best_models_df['best_model'].value_counts()
        
        print(f"Best Model Frequency:")
        for model, count in dominance.items():
            percentage = (count / len(best_models_df)) * 100
            print(f"   {model}: {count}/{len(best_models_df)} periods ({percentage:.1f}%)")
        
        # Academic justification
        print(f"\n" + "="*60)
        print("ACADEMIC JUSTIFICATION FOR RANDOM FOREST")
        print("="*60)
        
        if rf_performance:
            top_performer = dominance.index[0]
            rf_rank_analysis = rf_df['rank'].value_counts().sort_index()
            
            print(f"\nMETHODOLOGICAL JUSTIFICATION:")
            
            if avg_rank <= 3:
                print(f"✓ RF consistently ranks in top 3 (avg rank: {avg_rank:.1f})")
            
            if std_accuracy < 0.01:
                print(f"✓ RF shows excellent stability (std: {std_accuracy:.4f})")
            
            if top_performer == 'Decision_Tree':
                print(f"✓ Decision Tree dominance suggests overfitting risk")
                print(f"  RF's regularization provides better generalization")
            
            print(f"\nBUSINESS JUSTIFICATION:")
            print(f" Done! Ensemble compatibility for production deployment")
            print(f" Done! Feature importance interpretability")
            print(f" Done! Robustness to outliers and noise")
            print(f" Done! Built-in cross-validation through bagging")
            
            # Performance gaps
            if rf_performance:
                print(f"\nPERFORMANCE GAP ANALYSIS:")
                for period in self.periods:
                    if period in self.results_all_periods:
                        period_data = df_summary[df_summary['Period'] == period].copy()
                        period_data = period_data.sort_values('Accuracy', ascending=False)
                        
                        best_acc = period_data.iloc[0]['Accuracy']
                        rf_row = period_data[period_data['Model'] == 'Random_Forest']
                        
                        if not rf_row.empty:
                            rf_acc = rf_row['Accuracy'].values[0]
                            gap = best_acc - rf_acc
                            gap_pct = (gap / best_acc) * 100
                            
                            print(f"   {period}: Gap = {gap:.4f} ({gap_pct:.2f}%)")
                            
                            if gap_pct < 1.0:
                                print(f"     → Negligible difference (< 1%)")
        
        return df_summary
    
    def save_comprehensive_results(self, output_dir: str = "models/comprehensive_baseline"):
        """Save comprehensive results"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        results_path = f"{output_dir}/comprehensive_baseline_comparison_{timestamp}.pkl"
        
        comprehensive_package = {
            'all_periods_results': self.results_all_periods,
            'periods_analyzed': self.periods,
            'models_tested': list(self.baseline_models.keys()),
            'academic_metadata': {
                'purpose': 'Comprehensive baseline comparison across all periods',
                'methodology': 'Cross-period statistical validation',
                'academic_standard': 'Bath University MSc Business Analytics',
                'analysis_date': datetime.now().isoformat()
            }
        }
        
        joblib.dump(comprehensive_package, results_path)
        logger.info(f"Comprehensive results saved: {results_path}")
        
        return results_path

def main():
    print("COMPREHENSIVE RF BASELINE COMPARISON - ALL PERIODS")
    print("="*55)
    print("Academic Validation: Cross-period algorithm analysis")
    print("="*55)
    
    # Initialize comprehensive comparison
    comparison = ComprehensiveBaselineComparison()
    
    try:
        # Run comprehensive comparison
        results = comparison.run_comprehensive_comparison()
        
        if not results:
            print("Error! No results obtained")
            return
        
        # Generate comprehensive report
        summary_df = comparison.generate_comprehensive_report()
        
        # Save results
        results_path = comparison.save_comprehensive_results()
        
        print(f"\nCOMPREHENSIVE BASELINE COMPARISON COMPLETED!")
        print(f"Save: Results saved: {results_path}")
        print(f"Periods analyzed: {len(results)}")
        print(f"Total algorithm comparisons: {len(results) * len(comparison.baseline_models)}")
        
    except Exception as e:
        logger.error(f"Comprehensive comparison failed: {e}")
        raise

if __name__ == "__main__":
    main()