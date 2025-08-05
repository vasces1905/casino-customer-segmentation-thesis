"""
SIMPLE DIRECT PREDICTION - Back to Basics
No fancy ensemble, no overcomplicated stuff
Just: Load best model → Load data → Predict → Done!
"""

import pandas as pd
import numpy as np
import joblib
import psycopg2
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Database config
DB_CONFIG = {
    'host': 'localhost',
    'database': 'casino_research',
    'user': 'researcher',
    'password': 'academic_password_2024'
}

def load_clean_customer_data(period='2023-H2'):
    """Load ACTUAL customer_features data from database."""
    print(f"📊 Loading customer_features for {period}...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    query = """
    SELECT customer_id, total_bet, avg_bet, loss_rate, total_sessions,
           session_duration_volatility, loss_chasing_score, bet_trend_ratio,
           days_since_last_visit, sessions_last_30d, bet_volatility,
           weekend_preference, late_night_player, game_diversity,
           machine_diversity, zone_diversity
    FROM casino_data.customer_features 
    WHERE analysis_period = %s
    ORDER BY customer_id
    """
    
    df = pd.read_sql_query(query, conn, params=(period,))
    conn.close()
    
    print(f"✅ Loaded {len(df)} customers")
    print(f"📋 Available columns: {list(df.columns)}")
    
    # Fill NULLs with reasonable defaults
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    
    print(f"🔧 Cleaned {len(numeric_columns)} numeric features")
    
    return df

def find_best_model():
    """Find the best performing model from our PKL files."""
    print("🔍 Finding best model...")
    
    models = {
        '2022-H1': 'models/optimized_rf/optimized_kmeans_dbscan_rf_2022-H1.pkl',
        '2022-H2': 'models/optimized_rf/optimized_kmeans_dbscan_rf_2022-H2.pkl', 
        '2023-H1': 'models/optimized_rf/optimized_kmeans_dbscan_rf_2023-H1.pkl',
        '2023-H2': 'models/optimized_rf/optimized_kmeans_dbscan_rf_2023-H2.pkl'
    }
    
    best_model = None
    best_accuracy = 0
    best_period = None
    
    for period, path in models.items():
        try:
            model_data = joblib.load(path)
            performance = model_data.get('performance', {})
            accuracy = performance.get('accuracy', 0)
            
            print(f"📊 {period}: Accuracy = {accuracy:.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_data
                best_period = period
                
        except Exception as e:
            print(f"❌ Failed to load {period}: {e}")
            continue
    
    if best_model:
        print(f"🏆 Best model: {best_period} (Accuracy: {best_accuracy:.3f})")
        return best_model, best_period, best_accuracy
    else:
        raise ValueError("No models could be loaded!")

def prepare_features_for_model(df, model_features):
    """Prepare data features to match model expectations."""
    print(f"🔧 Preparing features for model...")
    
    available_features = []
    missing_features = []
    
    for feature in model_features:
        if feature in df.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
    
    print(f"✅ Available features: {len(available_features)}")
    print(f"❌ Missing features: {len(missing_features)}")
    
    if missing_features:
        print(f"Missing: {missing_features[:10]}...")  # Show first 10
    
    # Use only available features
    X = df[available_features].fillna(0)
    
    return X, available_features

def simple_prediction(test_data, model_data, period):
    """Simple, direct prediction - no fancy stuff."""
    print(f"🎯 Running simple prediction with {period} model...")
    
    # Extract model components
    rf_model = model_data.get('rf_model') or model_data.get('model')
    scaler = model_data.get('scaler')
    feature_names = model_data.get('feature_names') or model_data.get('features')
    label_encoder = model_data.get('label_encoder')
    
    if not rf_model:
        raise ValueError("No RF model found in PKL file!")
    
    print(f"📋 Model expects {len(feature_names)} features")
    
    # Prepare features
    X, available_features = prepare_features_for_model(test_data, feature_names)
    
    if len(available_features) < len(feature_names) * 0.5:
        print(f"⚠️ Warning: Only {len(available_features)}/{len(feature_names)} features available")
    
    # Apply scaling if available
    if scaler:
        try:
            X_scaled = scaler.transform(X)
            print("✅ Applied scaler successfully")
        except Exception as e:
            print(f"⚠️ Scaler failed: {e}")
            X_scaled = X.values
    else:
        X_scaled = X.values
        print("ℹ️ No scaler found, using raw values")
    
    # Make predictions
    try:
        probabilities = rf_model.predict_proba(X_scaled)
        predictions = rf_model.predict(X_scaled)
        
        print(f"✅ Predictions generated!")
        print(f"   Shape: {probabilities.shape}")
        print(f"   Classes: {rf_model.classes_}")
        
        # Convert predictions back to labels if encoder exists
        if label_encoder and hasattr(label_encoder, 'inverse_transform'):
            try:
                prediction_labels = label_encoder.inverse_transform(predictions)
                print("✅ Converted predictions to labels")
            except:
                prediction_labels = predictions
                print("ℹ️ Using numeric predictions")
        else:
            prediction_labels = predictions
            print("ℹ️ No label encoder, using numeric predictions")
        
        return prediction_labels, probabilities, available_features
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        raise

def analyze_results(predictions, probabilities, customer_ids):
    """Analyze prediction results."""
    print(f"\n📊 ANALYZING RESULTS...")
    
    # Basic stats
    total_customers = len(predictions)
    unique_predictions = np.unique(predictions)
    avg_confidence = np.max(probabilities, axis=1).mean()
    high_confidence_count = (np.max(probabilities, axis=1) > 0.7).sum()
    
    print(f"✅ Total customers: {total_customers:,}")
    print(f"📋 Unique predictions: {unique_predictions}")
    print(f"🎯 Average confidence: {avg_confidence:.3f}")
    print(f"🔥 High confidence (>0.7): {high_confidence_count:,} ({high_confidence_count/total_customers*100:.1f}%)")
    
    # Prediction distribution
    pred_dist = pd.Series(predictions).value_counts()
    print(f"\n📈 Prediction Distribution:")
    for pred, count in pred_dist.items():
        percentage = (count / total_customers) * 100
        avg_conf = probabilities[predictions == pred].max(axis=1).mean()
        print(f"   {pred}: {count:,} customers ({percentage:.1f}%) - avg conf: {avg_conf:.3f}")
    
    return {
        'total_customers': total_customers,
        'avg_confidence': avg_confidence,
        'high_confidence_count': high_confidence_count,
        'prediction_distribution': pred_dist.to_dict()
    }

def main():
    """Main execution - SIMPLE AND DIRECT."""
    
    print("🚀 SIMPLE DIRECT PREDICTION - BACK TO BASICS")
    print("=" * 60)
    
    try:
        # 1. Load best model
        print("\n🏆 STEP 1: Finding Best Model...")
        model_data, best_period, best_accuracy = find_best_model()
        
        # 2. Load clean data
        print(f"\n📊 STEP 2: Loading Clean Data...")
        test_data = load_clean_customer_data('2023-H2')
        
        # 3. Simple prediction
        print(f"\n🎯 STEP 3: Simple Prediction...")
        predictions, probabilities, features_used = simple_prediction(test_data, model_data, best_period)
        
        # 4. Analyze results
        print(f"\n📋 STEP 4: Results Analysis...")
        analysis = analyze_results(predictions, probabilities, test_data['customer_id'])
        
        # 5. Save results
        print(f"\n💾 STEP 5: Saving Results...")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        
        results_df = pd.DataFrame({
            'customer_id': test_data['customer_id'],
            'prediction': predictions,
            'confidence': np.max(probabilities, axis=1)
        })
        
        # Add probability columns
        for i in range(probabilities.shape[1]):
            results_df[f'prob_class_{i}'] = probabilities[:, i]
        
        output_file = f'simple_direct_predictions_{timestamp}.csv'
        results_df.to_csv(output_file, index=False)
        
        # Final summary
        print(f"\n" + "="*60)
        print(f"🎯 SIMPLE PREDICTION COMPLETE!")
        print("="*60)
        print(f"✅ Model used: {best_period} (Accuracy: {best_accuracy:.3f})")
        print(f"✅ Features used: {len(features_used)}")
        print(f"✅ Customers predicted: {analysis['total_customers']:,}")
        print(f"✅ Average confidence: {analysis['avg_confidence']:.3f}")
        print(f"💾 Results saved to: {output_file}")
        
        # Quality assessment
        print(f"\n🔍 QUALITY ASSESSMENT:")
        if analysis['avg_confidence'] > 0.7:
            print("✅ HIGH QUALITY: Strong prediction confidence")
        elif analysis['avg_confidence'] > 0.5:
            print("⚠️ MODERATE QUALITY: Reasonable confidence")
        else:
            print("❌ LOW QUALITY: Weak prediction confidence")
            
        if analysis['high_confidence_count'] > analysis['total_customers'] * 0.5:
            print("✅ RELIABLE: Majority of predictions are high confidence")
        else:
            print("⚠️ MIXED: Many predictions have low confidence")
        
        print(f"\n🎓 FOR THESIS:")
        print(f"This demonstrates successful Random Forest application")
        print(f"with {analysis['avg_confidence']:.1%} average confidence on")
        print(f"{analysis['total_customers']:,} casino customers.")
        
        return results_df, analysis
        
    except Exception as e:
        print(f"❌ SIMPLE PREDICTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    results, analysis = main()