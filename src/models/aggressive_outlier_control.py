# aggressive_outlier_control_fixed.py - Fixed version without MaskedArray issues
import numpy as np
import pandas as pd
import psycopg2
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def winsorize_simple(data, limits=(0.005, 0.005)):
    """Simple winsorization without MaskedArray issues"""
    data_array = np.array(data)
    lower_limit = limits[0]
    upper_limit = 1 - limits[1]
    
    lower_val = np.percentile(data_array, lower_limit * 100)
    upper_val = np.percentile(data_array, upper_limit * 100)
    
    return np.clip(data_array, lower_val, upper_val)

def aggressive_outlier_removal(df):
    """Remove and cap extreme outliers - FIXED VERSION"""
    logger.info("🔧 PHASE 1: Aggressive Outlier Removal")
    
    df_clean = df.copy()
    original_count = len(df_clean)
    
    # STEP 1: Remove mega outliers (>€50M) entirely
    mega_outlier_mask = df_clean['total_bet'] > 50_000_000
    mega_outliers_removed = mega_outlier_mask.sum()
    df_clean = df_clean[~mega_outlier_mask].copy()
    
    # Track removed outliers for academic documentation
    df_clean['was_mega_outlier'] = False
    
    # STEP 2: Cap extreme outliers at 99.5th percentile by period
    outlier_caps = {}
    for period in df_clean['analysis_period'].unique():
        period_mask = df_clean['analysis_period'] == period
        period_data = df_clean[period_mask]
        
        # Calculate period-specific cap
        cap_995 = period_data['total_bet'].quantile(0.995)
        outlier_caps[period] = cap_995
        
        # Apply capping
        df_clean.loc[period_mask, 'total_bet_capped'] = period_data['total_bet'].clip(upper=cap_995)
        
        logger.info(f"   {period}: Cap at €{cap_995:,.2f} (99.5th percentile)")
    
    # STEP 3: Simple winsorization (FIXED - no MaskedArray)
    df_clean['total_bet_winsorized'] = winsorize_simple(
        df_clean['total_bet_capped'], 
        limits=(0.005, 0.005)  # Winsorize top/bottom 0.5%
    )
    
    # STEP 4: Log transformation for stability
    df_clean['log_bet_stable'] = np.log1p(df_clean['total_bet_winsorized'])
    
    logger.info(f"   ✅ Removed {mega_outliers_removed} mega outliers (>€50M)")
    logger.info(f"   ✅ Capped extreme values at 99.5th percentile by period")
    logger.info(f"   ✅ Applied simple winsorization and log transformation")
    
    return df_clean, outlier_caps

def period_stratified_features(df):
    """Create period-aware robust features - FIXED VERSION"""
    logger.info("🔧 PHASE 2: Period-Stratified Feature Engineering")
    
    df_features = df.copy()
    
    # Initialize columns to avoid warnings
    df_features['bet_percentile_in_period'] = 0.0
    df_features['risk_percentile_in_period'] = 0.0
    df_features['session_percentile_in_period'] = 0.0
    df_features['bet_zscore_in_period'] = 0.0
    
    # PERIOD-NORMALIZED FEATURES (combat 2023-H2 dominance)
    for period in df_features['analysis_period'].unique():
        period_mask = df_features['analysis_period'] == period
        period_data = df_features[period_mask]
        
        if len(period_data) > 1:  # Need at least 2 points for ranking
            # Percentile ranks within period (0-1 scale)
            df_features.loc[period_mask, 'bet_percentile_in_period'] = period_data['total_bet_winsorized'].rank(pct=True)
            df_features.loc[period_mask, 'risk_percentile_in_period'] = period_data['loss_chasing_score'].rank(pct=True)
            df_features.loc[period_mask, 'session_percentile_in_period'] = period_data['total_sessions'].rank(pct=True)
            
            # Z-scores within period (normalized to period mean/std)
            period_mean_bet = period_data['total_bet_winsorized'].mean()
            period_std_bet = period_data['total_bet_winsorized'].std()
            
            if period_std_bet > 0:
                df_features.loc[period_mask, 'bet_zscore_in_period'] = (
                    period_data['total_bet_winsorized'] - period_mean_bet
                ) / period_std_bet
    
    # ROBUST INTERACTION FEATURES
    df_features['robust_risk_score'] = (
        df_features['bet_percentile_in_period'] * 0.3 +
        df_features['risk_percentile_in_period'] * 0.4 +
        df_features['session_percentile_in_period'] * 0.2 +
        df_features['is_dbscan_outlier'].astype(int) * 0.1
    )
    
    # STABILITY INDICATORS
    df_features['is_stable_customer'] = (
        (df_features['bet_zscore_in_period'].abs() < 2.0) &  # Within 2 std
        (df_features['total_sessions'] >= 2) &  # Minimum activity
        (~df_features['was_mega_outlier'])  # Not mega outlier
    ).astype(int)
    
    logger.info(f"   ✅ Created period-normalized percentile features")
    logger.info(f"   ✅ Generated robust interaction scores")
    logger.info(f"   ✅ Added stability indicators")
    
    return df_features

def enhanced_sample_weighting(df):
    """Enhanced sample weights with outlier penalty - FIXED VERSION"""
    logger.info("🔧 PHASE 3: Enhanced Sample Weighting")
    
    df_weighted = df.copy()
    
    # BASE PERIOD WEIGHTS
    base_weights = {
        '2022-H1': 1.0,
        '2022-H2': 1.0,
        '2023-H1': 1.0,
        '2023-H2': 0.3  # Reduced from 0.4 to 0.3
    }
    
    # Calculate 99th percentile threshold for high bet detection
    high_bet_threshold = df_weighted['total_bet_winsorized'].quantile(0.99)
    
    # ASSIGN WEIGHTS
    enhanced_weights = []
    for _, row in df_weighted.iterrows():
        # Base period weight
        period_weight = base_weights.get(row['analysis_period'], 1.0)
        
        # Outlier penalty
        if row['was_mega_outlier']:
            outlier_weight = 0.1  # Mega outliers get minimal weight
        elif row['total_bet_winsorized'] > high_bet_threshold:
            outlier_weight = 0.8  # High bet customers get less weight
        elif row['is_dbscan_outlier']:
            outlier_weight = 0.7  # DBSCAN outliers get reduced weight
        else:
            outlier_weight = 1.0  # Normal customers
        
        # Combined weight
        final_weight = period_weight * outlier_weight
        enhanced_weights.append(final_weight)
    
    df_weighted['enhanced_sample_weight'] = enhanced_weights
    
    # LOG WEIGHT DISTRIBUTION
    weight_summary = df_weighted.groupby('analysis_period')['enhanced_sample_weight'].agg(['mean', 'min', 'max']).round(3)
    logger.info(f"   Enhanced Sample Weight Distribution:")
    for period in weight_summary.index:
        stats = weight_summary.loc[period]
        logger.info(f"   {period}: mean={stats['mean']}, min={stats['min']}, max={stats['max']}")
    
    return df_weighted

def create_promotion_labels_robust(df):
    """Create robust promotion labels with outlier handling - FIXED VERSION"""
    logger.info("🔧 PHASE 4: Robust Promotion Label Creation")
    
    df_labeled = df.copy()
    promotion_labels = []
    
    for _, customer in df_labeled.iterrows():
        # MEGA OUTLIER INTERVENTION (highest priority)
        if customer['was_mega_outlier']:
            label = 'MEGA_OUTLIER_INTERVENTION'
            
        # HIGH RISK INTERVENTION (DBSCAN + high risk score)
        elif (customer['is_dbscan_outlier'] or 
              customer['robust_risk_score'] > 0.8):
            label = 'HIGH_RISK_INTERVENTION'
            
        # VIP TREATMENT (high value + stable)
        elif (customer['kmeans_segment_encoded'] >= 3 and  # High_Value_Player
              customer['is_stable_customer'] and
              customer['bet_percentile_in_period'] > 0.9):
            label = 'VIP_TREATMENT'
            
        # GROWTH TARGET (regular + growth potential)
        elif (customer['kmeans_segment_encoded'] == 2 and  # Regular_Player
              customer['is_stable_customer'] and
              customer['bet_percentile_in_period'] > 0.7):
            label = 'GROWTH_TARGET'
            
        # STANDARD PROMOTION (casual + stable)
        elif (customer['is_stable_customer'] and
              customer['total_sessions'] >= 3 and
              customer['robust_risk_score'] < 0.5):
            label = 'STANDARD_PROMOTION'
            
        # NO PROMOTION (everyone else)
        else:
            label = 'NO_PROMOTION'
            
        promotion_labels.append(label)
    
    df_labeled['robust_promotion_label'] = promotion_labels
    
    # LOG DISTRIBUTION
    label_dist = df_labeled['robust_promotion_label'].value_counts()
    logger.info(f"   Robust Promotion Label Distribution:")
    for label, count in label_dist.items():
        pct = count / len(df_labeled) * 100
        logger.info(f"   {label}: {count} customers ({pct:.1f}%)")
    
    return df_labeled

def load_and_process_all_periods():
    """Load and process data for all periods with aggressive outlier control - FIXED"""
    logger.info("🚀 STARTING MAJOR FIX - AGGRESSIVE OUTLIER CONTROL (FIXED VERSION)")
    
    try:
        # Database connection
        conn = psycopg2.connect(
            host="localhost",
            database="casino_research", 
            user="researcher",
            password="academic_password_2024"
        )
        
        # Load all period data
        query = """
        SELECT 
            cf.customer_id,
            cf.analysis_period,
            cf.total_bet,
            cf.loss_chasing_score,
            cf.total_sessions,
            cf.loss_rate,
            cf.session_duration_volatility,
            cf.days_since_last_visit,
            cf.bet_trend_ratio,
            ks.cluster_label as kmeans_segment,
            ks.cluster_id as kmeans_cluster_id,
            COALESCE(mas.is_outlier, false) as is_dbscan_outlier
        FROM casino_data.customer_features cf
        INNER JOIN casino_data.kmeans_segments ks 
            ON cf.customer_id = ks.customer_id 
            AND ks.period_id = cf.analysis_period
            AND ks.created_at >= '2025-07-21'
        LEFT JOIN casino_data.multi_algorithm_segments mas
            ON cf.customer_id = mas.customer_id
            AND mas.period_id = cf.analysis_period
            AND mas.algorithm_name = 'dbscan'
        WHERE cf.analysis_period IN ('2022-H1', '2022-H2', '2023-H1', '2023-H2')
        AND cf.total_bet > 0
        ORDER BY cf.analysis_period, cf.customer_id
        """
        
        df_raw = pd.read_sql_query(query, conn)
        conn.close()
        
        logger.info(f"📊 Loaded {len(df_raw)} customers across all periods")
        
        # Encode segments
        segment_encoding = {
            'Casual_Player': 1,
            'Regular_Player': 2, 
            'High_Value_Player': 3,
            'At_Risk_Player': 0
        }
        df_raw['kmeans_segment_encoded'] = df_raw['kmeans_segment'].map(segment_encoding).fillna(0)
        
        # Fill any missing values
        df_raw = df_raw.fillna(0)
        
        # Apply all phases
        df_clean, outlier_caps = aggressive_outlier_removal(df_raw)
        df_features = period_stratified_features(df_clean)
        df_weighted = enhanced_sample_weighting(df_features)
        df_final = create_promotion_labels_robust(df_weighted)
        
        # Save processed data for next step
        import os
        os.makedirs('data', exist_ok=True)
        df_final.to_csv('data/df_processed_major_fix.csv', index=False)
        logger.info(f"✅ Processed data saved to: data/df_processed_major_fix.csv")
        
        return df_final, outlier_caps
        
    except Exception as e:
        logger.error(f"❌ Error in processing: {str(e)}")
        raise

def main():
    """Main processing pipeline - FIXED VERSION"""
    try:
        # Process all data with aggressive outlier control
        df_processed, outlier_caps = load_and_process_all_periods()
        
        logger.info("✅ MAJOR FIX PHASE 1 COMPLETED!")
        logger.info(f"   Final dataset: {len(df_processed)} customers")
        
        # Summary statistics
        print("\n📊 PROCESSING SUMMARY:")
        print(f"   Total customers processed: {len(df_processed):,}")
        
        # Period summary
        period_summary = df_processed.groupby('analysis_period').agg({
            'total_bet_winsorized': ['count', 'mean', 'max'],
            'was_mega_outlier': 'sum',
            'enhanced_sample_weight': 'mean'
        }).round(2)
        
        print("\n📋 PERIOD SUMMARY:")
        for period in period_summary.index:
            data = period_summary.loc[period]
            print(f"   {period}: {data[('total_bet_winsorized', 'count')]} customers, "
                  f"avg bet: €{data[('total_bet_winsorized', 'mean')]:,.2f}, "
                  f"max bet: €{data[('total_bet_winsorized', 'max')]:,.2f}")
        
        print("\n🎯 NEXT STEP: Run robust RF training")
        print("   Command: python robust_rf_training_major_fix.py")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ MAJOR FIX FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ MAJOR FIX COMPLETED SUCCESSFULLY!")
    else:
        print("\n❌ MAJOR FIX FAILED - CHECK LOGS")