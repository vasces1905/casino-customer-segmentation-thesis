#!/usr/bin/env python3
"""
2022-H1 Data Quality Check
Kontrol edelim ki 2022-H1 gerçekten aynı kalitede veri mi?
"""

import pandas as pd
import psycopg2
import numpy as np

def check_2022_h1_data():
    """2022-H1 veri kalitesini kontrol et"""
    
    conn = psycopg2.connect(
        host="localhost",
        database="casino_research", 
        user="researcher",
        password="academic_password_2024"
    )
    
    print("🔍 2022-H1 DATA QUALITY CHECK")
    print("=" * 50)
    
    # 1. Available periods check
    periods_query = """
    SELECT DISTINCT analysis_period, COUNT(*) as customer_count
    FROM casino_data.customer_features 
    GROUP BY analysis_period 
    ORDER BY analysis_period
    """
    
    periods_df = pd.read_sql(periods_query, conn)
    print("\n📅 Available periods in customer_features:")
    print(periods_df.to_string(index=False))
    
    # 2. Promo label periods
    promo_periods_query = """
    SELECT DISTINCT period, COUNT(*) as label_count
    FROM casino_data.promo_label 
    GROUP BY period 
    ORDER BY period
    """
    
    promo_df = pd.read_sql(promo_periods_query, conn)
    print("\n🏷️ Available periods in promo_label:")
    print(promo_df.to_string(index=False))
    
    # 3. 2022-H1 specific analysis
    if '2022-H1' in periods_df['analysis_period'].values:
        print("\n🎯 2022-H1 DETAILED ANALYSIS:")
        
        # Feature completeness
        h1_features_query = """
        SELECT 
            COUNT(*) as total_customers,
            COUNT(total_bet) as has_total_bet,
            COUNT(loss_chasing_score) as has_loss_chasing,
            COUNT(bet_trend_ratio) as has_bet_trend,
            AVG(total_bet) as avg_total_bet,
            AVG(loss_chasing_score) as avg_loss_chasing,
            MIN(total_bet) as min_bet,
            MAX(total_bet) as max_bet
        FROM casino_data.customer_features 
        WHERE analysis_period = '2022-H1'
        """
        
        h1_stats = pd.read_sql(h1_features_query, conn)
        print("\n📊 2022-H1 Feature Statistics:")
        for col in h1_stats.columns:
            print(f"   {col}: {h1_stats[col].iloc[0]}")
        
        # Compare with 2022-H2
        if '2022-H2' in periods_df['analysis_period'].values:
            h2_features_query = """
            SELECT 
                COUNT(*) as total_customers,
                COUNT(total_bet) as has_total_bet,
                COUNT(loss_chasing_score) as has_loss_chasing,
                COUNT(bet_trend_ratio) as has_bet_trend,
                AVG(total_bet) as avg_total_bet,
                AVG(loss_chasing_score) as avg_loss_chasing,
                MIN(total_bet) as min_bet,
                MAX(total_bet) as max_bet
            FROM casino_data.customer_features 
            WHERE analysis_period = '2022-H2'
            """
            
            h2_stats = pd.read_sql(h2_features_query, conn)
            print("\n📊 2022-H2 Feature Statistics (COMPARISON):")
            for col in h2_stats.columns:
                print(f"   {col}: {h2_stats[col].iloc[0]}")
            
            print("\n🔄 H1 vs H2 Comparison:")
            print(f"   Customer Count Ratio: {h1_stats['total_customers'].iloc[0] / h2_stats['total_customers'].iloc[0]:.2f}")
            print(f"   Avg Bet Ratio: {h1_stats['avg_total_bet'].iloc[0] / h2_stats['avg_total_bet'].iloc[0]:.2f}")
    
    # 4. JOIN quality check
    if '2022-H1' in promo_df['period'].values:
        join_quality_query = """
        SELECT 
            cf.analysis_period,
            COUNT(cf.customer_id) as customers_in_features,
            COUNT(pl.customer_id) as customers_in_labels,
            COUNT(CASE WHEN pl.customer_id IS NOT NULL THEN 1 END) as successful_joins
        FROM casino_data.customer_features cf
        LEFT JOIN casino_data.promo_label pl 
            ON cf.customer_id = pl.customer_id 
            AND cf.analysis_period = pl.period
        WHERE cf.analysis_period = '2022-H1'
        GROUP BY cf.analysis_period
        """
        
        join_df = pd.read_sql(join_quality_query, conn)
        print("\n🔗 2022-H1 JOIN Quality:")
        print(join_df.to_string(index=False))
        
        join_rate = join_df['successful_joins'].iloc[0] / join_df['customers_in_features'].iloc[0]
        print(f"   JOIN Success Rate: {join_rate:.2%}")
        
        if join_rate < 0.95:
            print("   ⚠️ WARNING: Low JOIN rate detected!")
        else:
            print("   ✅ JOIN rate looks good")
    
    # 5. Sample data comparison
    print("\n🔍 SAMPLE DATA COMPARISON:")
    
    for period in ['2022-H1', '2022-H2']:
        if period in periods_df['analysis_period'].values:
            sample_query = f"""
            SELECT 
                customer_id, total_bet, loss_chasing_score, 
                bet_trend_ratio, days_since_last_visit
            FROM casino_data.customer_features 
            WHERE analysis_period = '{period}'
            ORDER BY customer_id
            LIMIT 3
            """
            
            sample_df = pd.read_sql(sample_query, conn)
            print(f"\n📋 {period} Sample (3 rows):")
            print(sample_df.to_string(index=False))
    
    conn.close()
    
    print("\n" + "=" * 50)
    print("✅ DATA QUALITY CHECK COMPLETED")

if __name__ == "__main__":
    check_2022_h1_data()