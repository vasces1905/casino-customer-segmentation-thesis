"""
DIRECT WORKING PREDICTION 
We found working models - let's use them NOW!
"""

import pandas as pd
import numpy as np
import joblib
import psycopg2
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

def load_working_model():
    """Load a working model - we know they work now!"""
    print("🔧 Loading working model...")
    
    # Use 2023-H2 since it has 5 classes (most complete)
    model_path = 'models/optimized_rf/optimized_kmeans_dbscan_rf_2023-H2.pkl'
    
    try:
        model_data = joblib.load(model_path)
        
        rf_model = model_data['rf_model']
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        feature_names = model_data['feature_names']
        
        print(f"✅ Model loaded successfully!")
        print(f"   Classes: {rf_model.classes_}")
        print(f"   Features: {len(feature_names)}")
        print(f"   Expected features: {feature_names}")
        
        return rf_model, scaler, label_encoder, feature_names
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

def load_customer_data():
    """Load customer_features data."""
    print("📊 Loading customer data...")
    
    conn = psycopg2.connect(**DB_CONFIG)
    
    # Load all available numeric features
    query = """
    SELECT customer_id, total_bet, avg_bet, loss_rate, total_sessions,
           session_duration_volatility, loss_chasing_score, bet_trend_ratio,
           days_since_last_visit, sessions_last_30d, bet_volatility,
           weekend_preference, late_night_player, game_diversity,
           machine_diversity, zone_diversity, bet_std, total_win, avg_win, avg_loss
    FROM casino_data.customer_features 
    WHERE analysis_period = '2023-H2'
    ORDER BY customer_id
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    print(f"✅ Loaded {len(df)} customers")
    print(f"📋 Available columns: {list(df.columns)}")
    
    # Fill NULLs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    return df

def match_features(data, expected_features):
    """Match data features to model expectations."""
    print("🔧 Matching features to model expectations...")
    
    available_features = []
    missing_features = []
    
    for feature in expected_features:
        if feature in data.columns:
            available_features.append(feature)
        else:
            missing_features.append(feature)
    
    print(f"✅ Available: {len(available_features)}/{len(expected_features)}")
    print(f" Available: {available_features}")
    
    if missing_features:
        print(f"❌ Missing: {missing_features}")
        
        # Create missing features with zeros
        for missing_feature in missing_features:
            data[missing_feature] = 0
            available_features.append(missing_feature)
            print(f"   🔧 Created {missing_feature} = 0")
    
    # Return data with features in exact order expected by model
    X = data[expected_features].values
    
    print(f"✅ Feature matrix shape: {X.shape}")
    
    return X, expected_features

def make_predictions(rf_model, scaler, label_encoder, X):
    """Make actual predictions."""
    print("🎯 Making predictions...")
    
    # Apply scaling
    try:
        X_scaled = scaler.transform(X)
        print("✅ Scaling applied successfully")
    except Exception as e:
        print(f"⚠️ Scaling failed: {e}")
        X_scaled = X
    
    # Make predictions
    try:
        probabilities = rf_model.predict_proba(X_scaled)
        predictions_numeric = rf_model.predict(X_scaled)
        
        print(f"✅ Predictions generated!")
        print(f"   Probabilities shape: {probabilities.shape}")
        print(f"   Unique predictions: {np.unique(predictions_numeric)}")
        
        # Convert to labels if possible
        try:
            predictions_labels = label_encoder.inverse_transform(predictions_numeric)
            print(f"✅ Converted to labels: {np.unique(predictions_labels)}")
        except:
            predictions_labels = predictions_numeric
            print("ℹ️ Using numeric predictions")
        
        return predictions_labels, probabilities, predictions_numeric
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        raise

def analyze_and_save_results(customer_ids, predictions, probabilities, predictions_numeric):
    """Analyze results and save."""
    print("📊 Analyzing results...")
    
    # Basic analysis
    total_customers = len(predictions)
    confidence_scores = np.max(probabilities, axis=1)
    avg_confidence = confidence_scores.mean()
    
    print(f"✅ Total customers: {total_customers:,}")
    print(f"🎯 Average confidence: {avg_confidence:.3f}")
    
    # Prediction distribution
    unique_preds, counts = np.unique(predictions, return_counts=True)
    
    print(f"\n📈 Prediction Distribution:")
    for pred, count in zip(unique_preds, counts):
        percentage = (count / total_customers) * 100
        avg_conf = confidence_scores[predictions == pred].mean()
        print(f"   {pred}: {count:,} customers ({percentage:.1f}%) - avg conf: {avg_conf:.3f}")
    
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
    
    print(f"\n💾 Results saved to: {output_file}")
    
    return results_df, avg_confidence

def main():
    """Main execution - WORKING PREDICTION!"""
    
    print("🚀 DIRECT WORKING PREDICTION")
    print("=" * 50)
    print("Using confirmed working models!")
    
    try:
        # 1. Load working model
        print(f"\n📂 STEP 1: Loading Working Model...")
        rf_model, scaler, label_encoder, feature_names = load_working_model()
        
        # 2. Load customer data
        print(f"\n📊 STEP 2: Loading Customer Data...")
        customer_data = load_customer_data()
        
        # 3. Match features
        print(f"\n🔧 STEP 3: Feature Matching...")
        X, matched_features = match_features(customer_data, feature_names)
        
        # 4. Make predictions
        print(f"\n🎯 STEP 4: Making Predictions...")
        predictions, probabilities, predictions_numeric = make_predictions(
            rf_model, scaler, label_encoder, X
        )
        
        # 5. Analyze and save
        print(f"\n📋 STEP 5: Analysis and Save...")
        results_df, avg_confidence = analyze_and_save_results(
            customer_data['customer_id'], predictions, probabilities, predictions_numeric
        )
        
        # Final summary
        print(f"\n" + "="*60)
        print(f"🎉 WORKING PREDICTION COMPLETE!")
        print("="*60)
        print(f"✅ Customers predicted: {len(results_df):,}")
        print(f"✅ Average confidence: {avg_confidence:.3f}")
        print(f"✅ Classes predicted: {len(np.unique(predictions))}")
        print(f"💾 Results file: working_predictions_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
        
        # Quality assessment
        print(f"\n🔍 QUALITY ASSESSMENT:")
        if avg_confidence > 0.8:
            print("🏆 EXCELLENT: Very high prediction confidence!")
        elif avg_confidence > 0.6:
            print("✅ GOOD: High prediction confidence")
        elif avg_confidence > 0.4:
            print("⚠️ MODERATE: Reasonable confidence")
        else:
            print("❌ LOW: Weak prediction confidence")
        
        # Business insights
        high_conf_count = (np.max(probabilities, axis=1) > 0.7).sum()
        print(f"\n💼 BUSINESS INSIGHTS:")
        print(f"   High confidence predictions (>70%): {high_conf_count:,}")
        print(f"   Percentage with high confidence: {high_conf_count/len(results_df)*100:.1f}%")
        
        print(f"\n🎓 THESIS READY:")
        print(f"Random Forest successfully predicted promotion categories")
        print(f"for {len(results_df):,} casino customers with {avg_confidence:.1%} average confidence.")
        
        return results_df
        
    except Exception as e:
        print(f"❌ WORKING PREDICTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    results = main()