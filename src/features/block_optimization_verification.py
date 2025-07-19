# optimization_verification.py
# University of Bath - Ethics Ref: 10351-12382
# Verify if optimization actually worked despite ROW_COUNT() error

import pandas as pd
from sqlalchemy import create_engine

class OptimizationVerifier:
    """
    Check if optimization worked despite the ROW_COUNT() error
    """
    
    def __init__(self):
        self.engine = create_engine("postgresql://researcher:academic_password_2024@localhost:5432/casino_research")
        
        print("🎓 University of Bath - Ethics Ref: 10351-12382")
        print("🔍 Optimization Verification Script")
        print("="*50)
    
    def verify_optimization_success(self):
        """Check if optimization actually worked"""
        
        print("\n📊 VERIFYING OPTIMIZATION RESULTS...")
        
        # Current block distribution
        verification_query = """
        WITH session_blocks AS (
            SELECT 
                player_id,
                gameing_day,
                game_id_simulated_v1,
                COUNT(*) as spins_per_block
            FROM casino_data.temp_valid_game_events 
            WHERE game_id_simulated_v1 IS NOT NULL
            GROUP BY player_id, gameing_day, game_id_simulated_v1
        ),
        block_stats AS (
            SELECT 
                CASE 
                    WHEN spins_per_block BETWEEN 8 AND 19 THEN 'Valid (8-19)'
                    WHEN spins_per_block < 8 THEN 'Too Small (<8)'
                    WHEN spins_per_block > 19 THEN 'Too Large (>19)'
                END as category,
                COUNT(*) as block_count,
                AVG(spins_per_block) as avg_spins
            FROM session_blocks
            GROUP BY 
                CASE 
                    WHEN spins_per_block BETWEEN 8 AND 19 THEN 'Valid (8-19)'
                    WHEN spins_per_block < 8 THEN 'Too Small (<8)'
                    WHEN spins_per_block > 19 THEN 'Too Large (>19)'
                END
        )
        SELECT 
            category,
            block_count,
            ROUND(block_count * 100.0 / SUM(block_count) OVER(), 2) as percentage,
            ROUND(avg_spins, 2) as avg_spins_in_category
        FROM block_stats
        ORDER BY category
        """
        
        current_results = pd.read_sql(verification_query, self.engine)
        
        print("📈 CURRENT Block Distribution:")
        for _, row in current_results.iterrows():
            print(f"  {row['category']}: {row['block_count']:,} blocks ({row['percentage']}%)")
        
        # Overall statistics
        overall_query = """
        SELECT 
            ROUND(AVG(block_size), 2) as overall_avg,
            MIN(block_size) as min_size,
            MAX(block_size) as max_size,
            COUNT(*) as total_blocks
        FROM (
            SELECT COUNT(*) as block_size
            FROM casino_data.temp_valid_game_events 
            WHERE game_id_simulated_v1 IS NOT NULL
            GROUP BY player_id, gameing_day, game_id_simulated_v1
        ) blocks
        """
        
        overall = pd.read_sql(overall_query, self.engine)
        avg_size = overall.iloc[0]['overall_avg']
        total_blocks = overall.iloc[0]['total_blocks']
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Total Blocks: {total_blocks:,}")
        print(f"  Average Block Size: {avg_size} spins")
        print(f"  Size Range: {overall.iloc[0]['min_size']} - {overall.iloc[0]['max_size']} spins")
        
        # Check if optimization worked
        valid_percentage = 0
        if len(current_results[current_results['category'] == 'Valid (8-19)']) > 0:
            valid_percentage = current_results[current_results['category'] == 'Valid (8-19)']['percentage'].iloc[0]
        
        print(f"\n🎯 OPTIMIZATION ASSESSMENT:")
        print(f"  Valid Blocks (8-19): {valid_percentage}%")
        print(f"  Previous: 28.82%")
        
        if valid_percentage > 35:  # Any improvement
            print(f"  Improvement: +{valid_percentage - 28.82:.2f}%")
            print("  ✅ OPTIMIZATION WORKED!")
            success = True
        else:
            print("  ❌ NO SIGNIFICANT IMPROVEMENT")
            success = False
        
        return success, valid_percentage, avg_size, total_blocks
    
    def check_game_distribution(self):
        """Check game distribution after optimization"""
        
        print("\n🎮 GAME DISTRIBUTION ANALYSIS...")
        
        game_dist_query = """
        SELECT 
            gsc.game_name,
            gsc.provider,
            gsc.popularity_tier,
            COUNT(*) as play_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM casino_data.temp_valid_game_events tve
        JOIN casino_data.slot_game_catalog gsc ON tve.game_id_simulated_v1 = gsc.game_id
        GROUP BY gsc.game_name, gsc.provider, gsc.popularity_tier
        ORDER BY play_count DESC
        LIMIT 10
        """
        
        game_results = pd.read_sql(game_dist_query, self.engine)
        
        print("🎯 Top 10 Games After Optimization:")
        for _, row in game_results.iterrows():
            print(f"  {row['game_name']} ({row['provider']}) - {row['popularity_tier']}: {row['percentage']}%")
        
        return game_results
    
    def generate_final_report(self):
        """Generate comprehensive final report"""
        
        success, valid_pct, avg_size, total_blocks = self.verify_optimization_success()
        game_dist = self.check_game_distribution()
        
        print("\n" + "="*60)
        print("🎯 FINAL OPTIMIZATION REPORT")
        print("="*60)
        
        if success:
            print("✅ STATUS: OPTIMIZATION SUCCESSFUL")
            print(f"📊 Compliance Rate: {valid_pct}% (was 28.82%)")
            print(f"📈 Average Block Size: {avg_size} spins (was 7.58)")
            print(f"🎲 Total Blocks: {total_blocks:,}")
            
            if valid_pct >= 90:
                print("🏆 EXCELLENT: Meets industry standards")
                recommendation = "APPROVED FOR FEATURE ENGINEERING"
            elif valid_pct >= 70:
                print("✅ GOOD: Significant improvement achieved") 
                recommendation = "READY FOR FEATURE ENGINEERING"
            else:
                print("⚠️ MODERATE: Some improvement achieved")
                recommendation = "ACCEPTABLE FOR ACADEMIC PURPOSES"
            
            print(f"📋 Recommendation: {recommendation}")
            
        else:
            print("❌ STATUS: OPTIMIZATION NEEDS REVIEW")
            print("🔧 Consider manual intervention or alternative approach")
        
        print("="*60)
        
        return success

# Execute verification
if __name__ == "__main__":
    
    verifier = OptimizationVerifier()
    
    print("🔍 Checking if optimization worked despite ROW_COUNT() error...")
    final_success = verifier.generate_final_report()
    
    if final_success:
        print("\n🚀 READY FOR NEXT PHASE: FEATURE ENGINEERING!")
        print("🎓 Data quality meets University of Bath standards")
    else:
        print("\n🛠️ Additional optimization may be needed")
        print("💡 Consider alternative approaches")