# ultra_fix_csv_saver.py - Save ultra-processed data to CSV
import numpy as np
import pandas as pd
import psycopg2
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_ultra_processed_data():
    """Re-run ultra-processing and save to CSV properly"""
    logger.info("🔧 SAVING ULTRA-PROCESSED DATA TO CSV")
    
    try:
        # Create data directory
        os.makedirs('data', exist_ok=True)
        
        # Database connection
        conn = psycopg2.connect(
            host="localhost",
            database="casino_research",
            user="researcher", 
            password="academic_password_2024"
        )
        
        # Load all data
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
        
        logger.info(f"📊 Loaded {len(df_raw)} customers")
        
        # Encode segments
        segment_encoding = {
            'Casual_Player': 1,
            'Regular_Player': 2,
            'High_Value_Player': 3,
            'At_Risk_Player': 0
        }
        df_raw['kmeans_segment_encoded'] = df_raw['kmeans_segment'].map(segment_encoding).fillna(0)
        df_raw = df_raw.fillna(0)
        
        # ULTRA-AGGRESSIVE PROCESSING (based on your successful terminal output)
        logger.info("🔥 APPLYING ULTRA-AGGRESSIVE PROCESSING")
        
        # STEP 1: Remove mega outliers (>€5M) - Terminal shows this worked!
        mega_outlier_mask = df_raw['total_bet'] > 5_000_000
        mega_removed = mega_outlier_mask.sum()
        df_ultra = df_raw[~mega_outlier_mask].copy()
        
        logger.info(f"   Removed {mega_removed} mega outliers (>€5M)")
        
        # STEP 2: Cap at 98th percentile by period (based on your terminal output)
        caps_applied = {}
        for period in df_ultra['analysis_period'].unique():
            period_mask = df_ultra['analysis_period'] == period
            period_data = df_ultra[period_mask]
            
            # 98th percentile cap
            cap_98 = period_data['total_bet'].quantile(0.98)
            caps_applied[period] = cap_98
            
            # Apply capping
            df_ultra.loc[period_mask, 'total_bet_ultra_capped'] = period_data['total_bet'].clip(upper=cap_98)
            
            logger.info(f"   {period}: Capped at €{cap_98:,.2f}")
        
        # STEP 3: Winsorization (1% limits)
        def winsorize_manual(data, limits=(0.01, 0.01)):
            data_array = np.array(data)
            lower_val = np.percentile(data_array, limits[0] * 100)
            upper_val = np.percentile(data_array, (1 - limits[1]) * 100)
            return np.clip(data_array, lower_val, upper_val)
        
        df_ultra['total_bet_ultra_winsorized'] = winsorize_manual(df_ultra['total_bet_ultra_capped'])
        
        # STEP 4: Log transformation
        df_ultra['log_bet_ultra'] = np.log1p(df_ultra['total_bet_ultra_winsorized'])
        
        # STEP 5: Period-normalized features
        for period in df_ultra['analysis_period'].unique():
            period_mask = df_ultra['analysis_period'] == period
            period_data = df_ultra[period_mask]
            
            if len(period_data) > 5:
                df_ultra.loc[period_mask, 'bet_percentile_ultra'] = period_data['total_bet_ultra_winsorized'].rank(pct=True)
                df_ultra.loc[period_mask, 'risk_percentile_ultra'] = period_data['loss_chasing_score'].rank(pct=True)
                df_ultra.loc[period_mask, 'session_percentile_ultra'] = period_data['total_sessions'].rank(pct=True)
        
        # STEP 6: Ultra-sample weights
        base_weights = {'2022-H1': 1.0, '2022-H2': 1.0, '2023-H1': 1.0, '2023-H2': 0.1}
        df_ultra['ultra_sample_weight'] = df_ultra['analysis_period'].map(base_weights)
        
        # STEP 7: Ultra-promotion labels
        promotion_labels = []
        for _, customer in df_ultra.iterrows():
            if customer['total_bet_ultra_winsorized'] > 1_000_000:
                label = 'HIGH_VALUE_INTERVENTION'
            elif customer['is_dbscan_outlier']:
                label = 'RISK_INTERVENTION'
            elif customer['kmeans_segment_encoded'] >= 3:
                label = 'VIP_TREATMENT'
            elif customer['total_sessions'] >= 3:
                label = 'STANDARD_PROMOTION'
            else:
                label = 'NO_PROMOTION'
            promotion_labels.append(label)
        
        df_ultra['ultra_promotion_label'] = promotion_labels
        
        # Add processing metadata
        df_ultra['processing_type'] = 'ultra_aggressive_fix'
        df_ultra['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df_ultra['mega_outliers_removed'] = mega_removed
        df_ultra['was_mega_outlier'] = False  # All mega outliers removed
        
        # SAVE TO CSV with proper path
        csv_path = 'data/df_ultra_aggressive_fix.csv'
        df_ultra.to_csv(csv_path, index=False)
        
        logger.info(f"✅ Ultra-processed data saved to: {csv_path}")
        
        # Validation summary
        mega_check = (df_ultra['total_bet_ultra_winsorized'] > 5_000_000).sum()
        max_bet = df_ultra['total_bet_ultra_winsorized'].max()
        cv = df_ultra['total_bet_ultra_winsorized'].std() / df_ultra['total_bet_ultra_winsorized'].mean()
        
        print(f"\n🎯 ULTRA-PROCESSED DATA VALIDATION:")
        print(f"   Mega outliers (>€5M): {mega_check}")
        print(f"   Max bet: €{max_bet:,.2f}")
        print(f"   Coefficient of Variation: {cv:.3f}")
        print(f"   Total customers: {len(df_ultra):,}")
        print(f"   File saved: {csv_path}")
        
        # Verify file exists
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path) / (1024*1024)  # MB
            print(f"   File size: {file_size:.1f} MB")
            print(f"   ✅ CSV FILE SUCCESSFULLY CREATED!")
        else:
            print(f"   ❌ CSV FILE NOT FOUND!")
        
        return df_ultra, csv_path
        
    except Exception as e:
        logger.error(f"❌ Error saving ultra-processed data: {str(e)}")
        return None, None

def main():
    """Main execution"""
    print("🔧 ULTRA-AGGRESSIVE FIX CSV SAVER")
    print("=" * 50)
    
    df_processed, csv_path = save_ultra_processed_data()
    
    if df_processed is not None:
        print("\n✅ ULTRA-PROCESSED DATA SAVED SUCCESSFULLY!")
        print(f"📁 Location: {csv_path}")
        print(f"📊 Customers: {len(df_processed):,}")
        print("\n🎯 READY FOR VALIDATION QUERIES!")
        print("   Now run validation queries on ultra-processed data")
    else:
        print("\n❌ FAILED TO SAVE ULTRA-PROCESSED DATA!")

if __name__ == "__main__":
    main()