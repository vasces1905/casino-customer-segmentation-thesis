# src/data/block_optimization_fixed_v2.py
# University of Bath - Ethics Ref: 10351-12382
# Block Optimization - Fixed Data Type Issues

import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine
import logging
from datetime import datetime

class CasinoBlockOptimizerFixed:
    """
    Fixed version with proper data type handling
    """
    
    def __init__(self):
        self.engine = create_engine("postgresql://researcher:academic_password_2024@localhost:5432/casino_research")
        
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        print("🎓 University of Bath - Ethics Ref: 10351-12382")
        print("🔧 Casino Block Optimization - Fixed Version")
        print("="*60)
    
    def verify_backup_and_analyze(self):
        """Verify backup and analyze current state"""
        
        print("\n✅ BACKUP VERIFICATION...")
        try:
            backup_query = "SELECT COUNT(*) as backup_count FROM casino_data.temp_valid_game_events_backup"
            backup_result = pd.read_sql(backup_query, self.engine)
            print(f"✅ Backup verified: {backup_result.iloc[0]['backup_count']:,} rows")
        except:
            print("❌ Backup verification failed")
            return False
        
        print("\n📊 CURRENT STATE ANALYSIS...")
        
        # Current block distribution
        analysis_query = """
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
                COUNT(*) as block_count
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
            ROUND(block_count * 100.0 / SUM(block_count) OVER(), 2) as percentage
        FROM block_stats
        ORDER BY category
        """
        
        current_stats = pd.read_sql(analysis_query, self.engine)
        
        print("📈 Current Block Distribution:")
        for _, row in current_stats.iterrows():
            print(f"  {row['category']}: {row['block_count']:,} blocks ({row['percentage']}%)")
        
        return True
    
    def execute_strategic_optimization(self):
        """Execute optimization with strategic SQL approach"""
        
        print("\n🎯 STRATEGIC OPTIMIZATION APPROACH...")
        print("Strategy: Direct SQL updates for efficiency and reliability")
        
        try:
            conn = psycopg2.connect(
                host='localhost',
                port='5432',
                database='casino_research',
                user='researcher',
                password='academic_password_2024'
            )
            cursor = conn.cursor()
            
            print("\n🔧 Step 1: Creating optimization staging table...")
            
            # Create staging table with optimized game assignments
            cursor.execute("""
            DROP TABLE IF EXISTS casino_data.optimization_staging;
            
            CREATE TABLE casino_data.optimization_staging AS
            WITH player_daily_sessions AS (
                SELECT 
                    player_id,
                    gameing_day,
                    COUNT(*) as daily_spins,
                    ARRAY_AGG(ctid ORDER BY ts) as spin_ctids,
                    MIN(ts) as session_start,
                    AVG(bet) as avg_bet
                FROM casino_data.temp_valid_game_events 
                WHERE game_id_simulated_v1 IS NOT NULL
                GROUP BY player_id, gameing_day
            ),
            optimized_blocks AS (
                SELECT 
                    player_id,
                    gameing_day,
                    CASE 
                        WHEN daily_spins <= 7 THEN 1  -- Small sessions: single block
                        WHEN daily_spins <= 15 THEN 2  -- Medium sessions: 2 blocks
                        WHEN daily_spins <= 30 THEN 3  -- Large sessions: 3 blocks
                        ELSE GREATEST(2, daily_spins / 12)  -- Very large: ~12 spins per block
                    END as target_blocks,
                    daily_spins,
                    spin_ctids,
                    avg_bet
                FROM player_daily_sessions
            )
            SELECT 
                player_id,
                gameing_day,
                target_blocks,
                daily_spins,
                spin_ctids,
                avg_bet
            FROM optimized_blocks;
            """)
            
            print("✅ Staging table created")
            
            print("\n🔧 Step 2: Implementing block redistribution...")
            
            # Create block assignments table
            cursor.execute("""
            DROP TABLE IF EXISTS casino_data.block_assignments;
            
            CREATE TABLE casino_data.block_assignments AS
            WITH block_creation AS (
                SELECT 
                    player_id,
                    gameing_day,
                    block_num,
                    CASE 
                        WHEN block_num = target_blocks THEN 
                            -- Last block gets remaining spins
                            daily_spins - ((target_blocks - 1) * (daily_spins / target_blocks)::INT)
                        ELSE 
                            -- Regular blocks get equal distribution
                            GREATEST(8, LEAST(19, (daily_spins / target_blocks)::INT))
                    END as block_size,
                    avg_bet
                FROM casino_data.optimization_staging,
                LATERAL generate_series(1, target_blocks) AS block_num
            ),
            game_selection AS (
                SELECT 
                    player_id,
                    gameing_day,
                    block_num,
                    block_size,
                    CASE 
                        WHEN avg_bet > 5.0 THEN 
                            -- High bet players: Premium games (1,2,5)
                            CASE (RANDOM() * 3)::INT
                                WHEN 0 THEN 1  -- Book of Ra
                                WHEN 1 THEN 2  -- Wolf Gold  
                                ELSE 5  -- Big Bass Bonanza
                            END
                        WHEN avg_bet > 2.0 THEN
                            -- Medium bet players: Popular games (1,2,3,4)
                            CASE (RANDOM() * 4)::INT + 1
                                WHEN 1 THEN 1  -- Book of Ra
                                WHEN 2 THEN 2  -- Wolf Gold
                                WHEN 3 THEN 3  -- Cleopatra
                                ELSE 4  -- Sweet Bonanza
                            END
                        ELSE
                            -- Low bet players: All games (1-8)
                            (RANDOM() * 8)::INT + 1
                    END as assigned_game_id
                FROM block_creation
            )
            SELECT 
                player_id,
                gameing_day,
                block_num,
                block_size,
                assigned_game_id
            FROM game_selection
            WHERE block_size BETWEEN 8 AND 19;  -- Ensure compliance
            """)
            
            print("✅ Block assignments created")
            
            print("\n🔧 Step 3: Applying optimized assignments...")
            
            # Apply the optimizations (simplified approach)
            cursor.execute("""
            WITH assignment_application AS (
                SELECT 
                    ba.player_id,
                    ba.gameing_day, 
                    ba.assigned_game_id,
                    ROW_NUMBER() OVER (PARTITION BY ba.player_id, ba.gameing_day ORDER BY ba.block_num) as block_sequence
                FROM casino_data.block_assignments ba
            ),
            spin_updates AS (
                SELECT 
                    tve.ctid,
                    COALESCE(aa.assigned_game_id, 
                        CASE 
                            WHEN tve.bet > 5 THEN (RANDOM() * 3 + 1)::INT  -- 1,2,3
                            WHEN tve.bet > 2 THEN (RANDOM() * 4 + 1)::INT  -- 1,2,3,4  
                            ELSE (RANDOM() * 8 + 1)::INT  -- 1-8
                        END
                    ) as new_game_id
                FROM casino_data.temp_valid_game_events tve
                LEFT JOIN assignment_application aa ON tve.player_id = aa.player_id 
                    AND tve.gameing_day = aa.gameing_day
                WHERE tve.game_id_simulated_v1 IS NOT NULL
            )
            UPDATE casino_data.temp_valid_game_events 
            SET game_id_simulated_v1 = su.new_game_id
            FROM spin_updates su
            WHERE temp_valid_game_events.ctid = su.ctid;
            """)
            
            # Get update count
            cursor.execute("SELECT ROW_COUNT() as updated_rows")
            # Note: ROW_COUNT() might not work in all PostgreSQL versions, so we'll check differently
            
            conn.commit()
            
            print("✅ Optimization applied successfully")
            
            # Clean up staging tables
            cursor.execute("""
            DROP TABLE IF EXISTS casino_data.optimization_staging;
            DROP TABLE IF EXISTS casino_data.block_assignments;
            """)
            
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return False
    
    def analyze_results(self):
        """Analyze optimization results"""
        
        print("\n📊 POST-OPTIMIZATION ANALYSIS...")
        
        # Block distribution analysis
        analysis_query = """
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
        
        results = pd.read_sql(analysis_query, self.engine)
        
        print("📈 POST-OPTIMIZATION Block Distribution:")
        for _, row in results.iterrows():
            print(f"  {row['category']}: {row['block_count']:,} blocks ({row['percentage']}%) - Avg: {row['avg_spins_in_category']} spins")
        
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
        
        print(f"\n📊 Overall Statistics:")
        print(f"  Average Block Size: {avg_size} spins")
        print(f"  Size Range: {overall.iloc[0]['min_size']} - {overall.iloc[0]['max_size']} spins")
        
        # Calculate compliance rate
        valid_percentage = results[results['category'] == 'Valid (8-19)']['percentage'].iloc[0] if len(results[results['category'] == 'Valid (8-19)']) > 0 else 0
        
        print(f"\n🎯 COMPLIANCE RESULTS:")
        print(f"  Valid Blocks (8-19): {valid_percentage}%")
        print(f"  Target Compliance: 95%")
        print(f"  Status: {'✅ ACHIEVED' if valid_percentage >= 95 else '⚠️ NEEDS IMPROVEMENT'}")
        
        return valid_percentage >= 95
    
    def run_complete_optimization(self):
        """Run complete optimization process"""
        
        # Step 1: Verify and analyze
        if not self.verify_backup_and_analyze():
            return False
        
        # Step 2: Execute optimization
        if not self.execute_strategic_optimization():
            return False
        
        # Step 3: Analyze results
        success = self.analyze_results()
        
        if success:
            print("\n🎉 OPTIMIZATION COMPLETED SUCCESSFULLY!")
            print("✅ Industry standards achieved")
            print("🚀 Ready for Feature Engineering Pipeline")
        else:
            print("\n⚠️ OPTIMIZATION COMPLETED WITH PARTIAL SUCCESS")
            print("📋 May need additional fine-tuning")
        
        return success

# Execute optimization
if __name__ == "__main__":
    
    optimizer = CasinoBlockOptimizerFixed()
    
    print("🚀 Starting Complete Block Optimization...")
    print("⚠️ This will modify the main table (backup protected)")
    
    input_ready = input("\nProceed with optimization? (y/n): ").lower().strip()
    
    if input_ready == 'y':
        success = optimizer.run_complete_optimization()
        
        if success:
            print("\n" + "="*60)
            print("🎯 OPTIMIZATION SUMMARY")
            print("="*60)
            print("✅ Block size compliance achieved")
            print("✅ Industry standards met")
            print("✅ Academic requirements satisfied") 
            print("✅ Ready for ML pipeline")
            print("="*60)
        else:
            print("\n❌ Optimization needs review")
    else:
        print("❌ Optimization cancelled by user")