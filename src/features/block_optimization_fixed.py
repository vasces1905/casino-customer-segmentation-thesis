# src/data/block_optimization_fixed.py
# University of Bath - Ethics Ref: 10351-12382
# Block Optimization for Industry Standards Compliance
# v2

import pandas as pd
import numpy as np
import psycopg2
from typing import Dict, List, Tuple
import logging
from datetime import datetime

class CasinoBlockOptimizer:
    """
    Optimizes game block assignments to meet industry standards:
    - Target: 8-19 spins per game (95%+ compliance)
    - Average: ~12 spins per game  
    - Realistic casino session patterns
    """
    
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.target_compliance = 95.0
        self.industry_standards = {
            'min_spins': 8,
            'max_spins': 19,
            'target_avg': 12,
            'preferred_range': [10, 11, 12, 13, 14, 15],  # Most common
            'extended_range': [8, 9, 16, 17, 18, 19]      # Less common
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def run_full_optimization(self) -> Dict:
        """Execute complete block optimization process"""
        
        self.logger.info("🎯 Starting Casino Block Optimization")
        self.logger.info("📊 Target: Industry Standards (8-19 spins, 95%+ compliance)")
        
        # Step 1: Analyze current state
        current_analysis = self._analyze_current_blocks()
        self._print_analysis_report("BEFORE OPTIMIZATION", current_analysis)
        
        if current_analysis['compliance_rate'] >= self.target_compliance:
            self.logger.info("✅ Data already meets compliance standards!")
            return current_analysis
        
        # Step 2: Create backup
        self._create_backup_table()
        
        # Step 3: Execute optimization
        self.logger.info("🔧 Executing block optimization...")
        optimization_stats = self._execute_optimization()
        
        # Step 4: Analyze results
        final_analysis = self._analyze_current_blocks()
        self._print_analysis_report("AFTER OPTIMIZATION", final_analysis)
        
        # Step 5: Generate compliance report
        report = self._generate_compliance_report(current_analysis, final_analysis, optimization_stats)
        
        self.logger.info("✅ Block optimization completed!")
        return report
    
    def _analyze_current_blocks(self) -> Dict:
        """Analyze current block size distribution"""
        
        query = """
        WITH block_analysis AS (
            SELECT 
                player_id,
                gaming_day,
                game_id_simulated_v1,
                COUNT(*) as spins_per_block
            FROM casino_data.temp_valid_game_events 
            WHERE game_id_simulated_v1 IS NOT NULL
            GROUP BY player_id, gaming_day, game_id_simulated_v1
        ),
        size_stats AS (
            SELECT 
                spins_per_block,
                COUNT(*) as block_count,
                CASE 
                    WHEN spins_per_block BETWEEN 8 AND 19 THEN 'Valid'
                    WHEN spins_per_block < 8 THEN 'Too_Small'
                    WHEN spins_per_block > 19 THEN 'Too_Large'
                END as category
            FROM block_analysis
            GROUP BY spins_per_block
        )
        SELECT 
            category,
            SUM(block_count) as total_blocks,
            ROUND(AVG(spins_per_block), 2) as avg_spins,
            MIN(spins_per_block) as min_spins,
            MAX(spins_per_block) as max_spins
        FROM size_stats ss
        JOIN (SELECT spins_per_block, category FROM size_stats GROUP BY spins_per_block, category) cat_map
        ON ss.spins_per_block = cat_map.spins_per_block
        GROUP BY category
        """
        
        conn = psycopg2.connect(**self.db_config)
        df = pd.read_sql(query, conn)
        conn.close()
        
        total_blocks = df['total_blocks'].sum()
        valid_blocks = df[df['category'] == 'Valid']['total_blocks'].sum() if 'Valid' in df['category'].values else 0
        
        # Overall statistics
        overall_query = """
        SELECT 
            ROUND(AVG(block_size), 2) as overall_avg,
            MIN(block_size) as overall_min,
            MAX(block_size) as overall_max,
            COUNT(*) as total_blocks
        FROM (
            SELECT COUNT(*) as block_size
            FROM casino_data.temp_valid_game_events 
            WHERE game_id_simulated_v1 IS NOT NULL
            GROUP BY player_id, gaming_day, game_id_simulated_v1
        ) blocks
        """
        
        conn = psycopg2.connect(**self.db_config)
        overall_df = pd.read_sql(overall_query, conn)
        conn.close()
        
        return {
            'total_blocks': int(total_blocks),
            'valid_blocks': int(valid_blocks),
            'compliance_rate': round(valid_blocks / total_blocks * 100, 2) if total_blocks > 0 else 0,
            'overall_avg': float(overall_df.iloc[0]['overall_avg']),
            'overall_min': int(overall_df.iloc[0]['overall_min']),
            'overall_max': int(overall_df.iloc[0]['overall_max']),
            'category_breakdown': df.to_dict('records')
        }
    
    def _create_backup_table(self):
        """Create backup of original data"""
        
        self.logger.info("💾 Creating backup table...")
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Drop existing backup if exists
        cursor.execute("DROP TABLE IF EXISTS casino_data.temp_valid_game_events_backup")
        
        # Create backup
        cursor.execute("""
        CREATE TABLE casino_data.temp_valid_game_events_backup AS 
        SELECT * FROM casino_data.temp_valid_game_events
        """)
        
        # Add backup metadata
        cursor.execute("""
        ALTER TABLE casino_data.temp_valid_game_events_backup 
        ADD COLUMN backup_created TIMESTAMP DEFAULT NOW()
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        self.logger.info("✅ Backup created: temp_valid_game_events_backup")
    
    def _execute_optimization(self) -> Dict:
        """Execute the main optimization algorithm"""
        
        # Load all data for processing
        query = """
        SELECT 
            ctid,
            player_id,
            ts,
            bet,
            win,
            gaming_day,
            game_id_simulated_v1
        FROM casino_data.temp_valid_game_events 
        WHERE game_id_simulated_v1 IS NOT NULL
        ORDER BY player_id, gaming_day, ts
        """
        
        conn = psycopg2.connect(**self.db_config)
        df = pd.read_sql(query, conn)
        conn.close()
        
        # Convert timestamp
        df['ts'] = pd.to_datetime(df['ts'])
        df['gaming_day'] = pd.to_datetime(df['gaming_day'])
        
        # Process optimization
        optimization_stats = {
            'players_processed': 0,
            'sessions_optimized': 0,
            'blocks_created': 0,
            'spins_reassigned': 0
        }
        
        optimized_assignments = []
        
        # Process by customer and day
        for (player_id, gaming_day), day_data in df.groupby(['player_id', 'gaming_day']):
            
            day_data = day_data.sort_values('ts')
            day_assignments = self._optimize_daily_session(day_data)
            
            optimized_assignments.extend(day_assignments)
            optimization_stats['players_processed'] += 1
            optimization_stats['sessions_optimized'] += 1
            optimization_stats['blocks_created'] += len(day_assignments)
            optimization_stats['spins_reassigned'] += len(day_data)
        
        # Apply optimizations to database
        self._apply_optimizations(optimized_assignments)
        
        return optimization_stats
    
    def _optimize_daily_session(self, day_data: pd.DataFrame) -> List[Dict]:
        """Optimize blocks for a single customer's daily session"""
        
        spins = day_data.to_dict('records')
        total_spins = len(spins)
        
        # Calculate optimal block distribution
        target_blocks = max(1, total_spins // 12)  # Target ~12 spins per block
        
        assignments = []
        current_spin_idx = 0
        
        for block_num in range(target_blocks):
            
            # Calculate spins for this block
            remaining_spins = total_spins - current_spin_idx
            remaining_blocks = target_blocks - block_num
            
            if remaining_blocks == 1:
                # Last block gets all remaining spins
                spins_for_block = remaining_spins
            else:
                # Calculate optimal size
                avg_remaining = remaining_spins / remaining_blocks
                spins_for_block = self._select_optimal_block_size(avg_remaining)
            
            # Ensure valid range
            spins_for_block = max(8, min(19, spins_for_block))
            spins_for_block = min(spins_for_block, remaining_spins)
            
            # Select game for this block
            block_spins = spins[current_spin_idx:current_spin_idx + spins_for_block]
            game_id = self._select_game_for_block(block_spins)
            
            # Create assignment
            assignments.append({
                'player_id': day_data.iloc[0]['player_id'],
                'gaming_day': day_data.iloc[0]['gaming_day'],
                'game_id': game_id,
                'spin_ctids': [spin['ctid'] for spin in block_spins],
                'block_size': len(block_spins)
            })
            
            current_spin_idx += spins_for_block
            
            if current_spin_idx >= total_spins:
                break
        
        return assignments
    
    def _select_optimal_block_size(self, target_size: float) -> int:
        """Select optimal block size based on industry standards"""
        
        # Preferred sizes (most realistic)
        preferred = self.industry_standards['preferred_range']
        extended = self.industry_standards['extended_range']
        
        # Find closest preferred size
        closest_preferred = min(preferred, key=lambda x: abs(x - target_size))
        
        # 80% chance for preferred, 20% for extended
        if np.random.random() < 0.8:
            return closest_preferred
        else:
            # Select from extended range
            if target_size < 10:
                return np.random.choice([8, 9])
            else:
                return np.random.choice([16, 17, 18, 19])
    
    def _select_game_for_block(self, block_spins: List[Dict]) -> int:
        """Select appropriate game for a block based on betting patterns"""
        
        if not block_spins:
            return 1  # Default to Book of Ra
        
        # Analyze betting pattern
        total_bet = sum(float(spin['bet']) for spin in block_spins)
        avg_bet = total_bet / len(block_spins)
        
        # Game selection logic based on bet size
        if avg_bet > 5.0:
            # High roller games
            games = [1, 2, 5]  # Book of Ra, Wolf Gold, Big Bass Bonanza
            weights = [0.4, 0.35, 0.25]
        elif avg_bet > 2.0:
            # Medium bet games  
            games = [1, 2, 3, 4]
            weights = [0.3, 0.3, 0.2, 0.2]
        else:
            # All games for low bets
            games = list(range(1, 9))
            weights = [0.2, 0.15, 0.15, 0.15, 0.1, 0.1, 0.1, 0.05]
        
        # Add time-based variance
        if block_spins:
            hour = pd.to_datetime(block_spins[0]['ts']).hour
            if 18 <= hour <= 22:  # Peak hours
                # Boost popular games (1 and 2)
                if 1 in games:
                    idx = games.index(1)
                    weights[idx] *= 1.2
                if 2 in games:
                    idx = games.index(2) 
                    weights[idx] *= 1.1
        
        # Normalize and select
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        return np.random.choice(games, p=weights)
    
    def _apply_optimizations(self, assignments: List[Dict]):
        """Apply optimized assignments to database"""
        
        self.logger.info(f"📝 Applying {len(assignments)} optimized assignments...")
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        total_updates = 0
        
        for assignment in assignments:
            ctid_list = "','".join(assignment['spin_ctids'])
            
            update_query = f"""
            UPDATE casino_data.temp_valid_game_events 
            SET game_id_simulated_v1 = %s
            WHERE ctid IN ('{ctid_list}')
            """
            
            cursor.execute(update_query, (assignment['game_id'],))
            total_updates += len(assignment['spin_ctids'])
        
        conn.commit()
        cursor.close()
        conn.close()
        
        self.logger.info(f"✅ Applied optimizations: {total_updates:,} spins updated")
    
    def _print_analysis_report(self, phase: str, analysis: Dict):
        """Print detailed analysis report"""
        
        print(f"\n{'='*60}")
        print(f"📊 BLOCK ANALYSIS REPORT - {phase}")
        print(f"{'='*60}")
        print(f"Total Blocks: {analysis['total_blocks']:,}")
        print(f"Valid Blocks (8-19): {analysis['valid_blocks']:,}")
        print(f"Compliance Rate: {analysis['compliance_rate']}%")
        print(f"Average Block Size: {analysis['overall_avg']} spins")
        print(f"Size Range: {analysis['overall_min']} - {analysis['overall_max']} spins")
        
        print(f"\n📈 Category Breakdown:")
        for category in analysis['category_breakdown']:
            print(f"  {category['category']}: {category['total_blocks']:,} blocks")
        
        if analysis['compliance_rate'] >= self.target_compliance:
            print(f"\n✅ MEETS INDUSTRY STANDARDS ({self.target_compliance}%+ compliance)")
        else:
            print(f"\n⚠️  BELOW TARGET ({self.target_compliance}% compliance needed)")
        
        print(f"{'='*60}\n")
    
    def _generate_compliance_report(self, before: Dict, after: Dict, stats: Dict) -> Dict:
        """Generate comprehensive compliance report"""
        
        improvement = after['compliance_rate'] - before['compliance_rate']
        
        report = {
            'optimization_success': after['compliance_rate'] >= self.target_compliance,
            'before_compliance': before['compliance_rate'],
            'after_compliance': after['compliance_rate'],
            'improvement': round(improvement, 2),
            'before_avg_size': before['overall_avg'],
            'after_avg_size': after['overall_avg'],
            'target_avg_size': self.industry_standards['target_avg'],
            'meets_industry_standards': after['compliance_rate'] >= self.target_compliance,
            'processing_stats': stats,
            'recommendation': 'APPROVED FOR FEATURE ENGINEERING' if after['compliance_rate'] >= self.target_compliance else 'NEEDS FURTHER OPTIMIZATION'
        }
        
        return report

# Main execution
if __name__ == "__main__":
    
    # Database configuration
    db_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'casino_research', 
        'user': 'researcher',
        'password': 'academic_password_2024'
    }
    
    print("🎯 University of Bath - Ethics Ref: 10351-12382")
    print("🔧 Casino Block Optimization - Industry Standards Compliance")
    print("=" * 70)
    
    # Initialize optimizer
    optimizer = CasinoBlockOptimizer(db_config)
    
    # Run optimization
    final_report = optimizer.run_full_optimization()
    
    # Print final summary
    print("\n" + "="*70)
    print("🎯 FINAL OPTIMIZATION SUMMARY")
    print("="*70)
    print(f"✅ Optimization Success: {final_report['optimization_success']}")
    print(f"📊 Compliance Improvement: +{final_report['improvement']}%")
    print(f"📈 Final Compliance Rate: {final_report['after_compliance']}%")
    print(f"🎲 Average Block Size: {final_report['after_avg_size']} spins")
    print(f"🏆 Meets Industry Standards: {final_report['meets_industry_standards']}")
    print(f"📋 Status: {final_report['recommendation']}")
    print("="*70)
    
    if final_report['optimization_success']:
        print("\n🚀 READY FOR FEATURE ENGINEERING PIPELINE!")
        print("🎓 Data quality meets University of Bath academic standards")
    else:
        print("\n⚠️ Additional optimization may be needed")
        print("💡 Consider manual review of edge cases")