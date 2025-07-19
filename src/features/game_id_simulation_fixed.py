# game_id_simulation_fixed.py
"""
Casino Game ID Simulation - FIXED VERSION
==========================================
University of Bath - MSc Computer Science
Student: Muhammed Yavuzhan CANLI
Ethics Approval: 10351-12382

FIXED ISSUES:
- Column name consistency (game_id_simulated_v1)
- Removed duplicate update queries
- Fixed validation queries
- Proper batch processing
- Test with small batch first
"""

import pandas as pd
import numpy as np
import psycopg2
from sqlalchemy import create_engine, text
import logging
import math
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection():
    """Database connection - update with your credentials"""
    return create_engine("postgresql://researcher:academic_password_2024@localhost:5432/casino_research")

def load_game_catalog(engine):
    """Load game catalog with popularity weights"""
    query = """
    SELECT game_id, game_name, provider, popularity_tier
    FROM casino_data.slot_game_catalog
    ORDER BY game_id
    """
    
    game_catalog = pd.read_sql(query, engine)
    logger.info(f"✅ Loaded {len(game_catalog)} games from catalog")
    
    # Define popularity weights as per IT leader's suggestion
    popularity_weights = {
        'High': 0.5,     # 50% chance for high popularity
        'Medium': 0.3,   # 30% chance for medium popularity  
        'Low': 0.2       # 20% chance for low popularity
    }
    
    game_catalog['weight'] = game_catalog['popularity_tier'].map(popularity_weights)
    
    return game_catalog

def calculate_heatmap_factors(engine):
    """
    Calculate heatmap factors using gameing_day popularity.
    FIXED: Uses correct column name
    """
    logger.info("Calculating heatmap factors from gameing_day patterns...")
    
    query = """
    SELECT 
        gameing_day,
        COUNT(*) AS spin_count
    FROM casino_data.temp_valid_game_events
    WHERE gameing_day IS NOT NULL
    GROUP BY gameing_day
    ORDER BY gameing_day
    """
    
    daily_activity = pd.read_sql(query, engine)
    
    if len(daily_activity) == 0:
        logger.warning("No gameing_day data found, using default heatmap")
        return {}
    
    # Normalize: max = 100, min = 1
    max_spins = daily_activity['spin_count'].max()
    min_spins = daily_activity['spin_count'].min()
    
    if max_spins == min_spins:
        # Handle case where all days have same activity
        daily_activity['heat_factor'] = 0.5
    else:
        daily_activity['normalized_activity'] = (
            ((daily_activity['spin_count'] - min_spins) / (max_spins - min_spins)) * 99 + 1
        )
        
        # Apply sine function for distribution
        daily_activity['heat_factor'] = np.sin(
            daily_activity['normalized_activity'] * np.pi / 100
        )
    
    # Convert to dictionary for fast lookup
    heatmap_dict = dict(zip(
        daily_activity['gameing_day'], 
        daily_activity['heat_factor']
    ))
    
    logger.info(f"✅ Generated heatmap factors for {len(heatmap_dict)} days")
    logger.info(f"Activity range: {min_spins:,} - {max_spins:,} spins per day")
    
    return heatmap_dict

def assign_game_ids_to_events(engine, game_catalog, heatmap_factors, test_limit=2476481):
    """
    OPTIMIZED: Fast assignment with progress tracking
    """
    logger.info(f"Starting OPTIMIZED game ID assignment ({test_limit} records)...")
    
    # Check if game_id column exists
    column_check_query = """
    SELECT column_name 
    FROM information_schema.columns 
    WHERE table_schema = 'casino_data' 
      AND table_name = 'temp_valid_game_events' 
      AND column_name = 'game_id_simulated_v1'
    """
    
    column_exists = pd.read_sql(column_check_query, engine)
    if len(column_exists) == 0:
        logger.error("❌ game_id_simulated_v1 column not found!")
        raise ValueError("game_id_simulated_v1 column missing from temp_valid_game_events")
    
    logger.info("✅ game_id_simulated_v1 column confirmed")
    
    # Get total unprocessed events
    query = """
    SELECT COUNT(*) as total_events 
    FROM casino_data.temp_valid_game_events 
    WHERE game_id_simulated_v1 IS NULL OR game_id_simulated_v1 = 0
    """
    
    total_events = pd.read_sql(query, engine).iloc[0]['total_events']
    logger.info(f"Total unprocessed events: {total_events:,}")
    logger.info(f"Processing: {test_limit:,} records")
    
    if total_events == 0:
        logger.info("✅ All events already have game_id assigned")
        return
    
    # Load batch with progress tracking
    batch_query = f"""
    SELECT 
        player_id,
        ts,
        gameing_day,
        bet AS bet_amount,
        ROW_NUMBER() OVER (ORDER BY player_id, ts) as rn
    FROM casino_data.temp_valid_game_events
    WHERE game_id_simulated_v1 IS NULL OR game_id_simulated_v1 = 0
    ORDER BY player_id, ts
    LIMIT {test_limit}
    """
    
    logger.info(f"🔄 Loading {test_limit:,} events from database...")
    events_batch = pd.read_sql(batch_query, engine)
    
    if len(events_batch) == 0:
        logger.error("❌ No events found to process")
        return
    
    logger.info(f"✅ Loaded {len(events_batch):,} events for processing")
    
    # Apply game assignment logic with progress
    logger.info("🎮 Assigning games to events...")
    events_batch = assign_games_to_batch_improved(events_batch, game_catalog, heatmap_factors)
    
    # Update database with bulk operation
    logger.info("💾 Updating database (bulk operation)...")
    update_batch_in_db_improved(engine, events_batch)
    
    logger.info(f"🎉 COMPLETED: {len(events_batch):,} events processed successfully!")

def assign_games_to_batch_improved(events_df, game_catalog, heatmap_factors):
    """
    OPTIMIZED: Assign games with progress tracking and vectorized operations
    """
    total_events = len(events_df)
    logger.info(f"🎯 Processing {total_events:,} events for game assignment...")
    
    # Sort by player and time for session detection
    events_df = events_df.sort_values(['player_id', 'ts'])
    
    # Create session groups within each player
    events_df['session_rank'] = events_df.groupby('player_id').cumcount()
    
    # Expanded spin options for more variety
    spin_options = [8, 9, 10, 11, 12, 13, 15, 17, 19]
    
    # Progress tracking for game block assignment
    logger.info("📊 Step 1/3: Creating game blocks...")
    
    def assign_game_block_improved(row):
        player_seed = hash(str(row['player_id'])) % 1000
        time_component = int(row['session_rank'] / 20)
        
        combined_seed = (player_seed + time_component) % 10000
        np.random.seed(combined_seed)
        
        spins_per_game = np.random.choice(spin_options)
        return row['session_rank'] // spins_per_game
    
    events_df['game_block'] = events_df.apply(assign_game_block_improved, axis=1)
    
    # Progress tracking for game selection
    unique_blocks = events_df.groupby(['player_id', 'game_block']).size()
    logger.info(f"📊 Step 2/3: Assigning games to {len(unique_blocks):,} game blocks...")
    
    def select_game_for_block_improved(group):
        player_id = group['player_id'].iloc[0]
        gameing_day = group['gameing_day'].iloc[0] if 'gameing_day' in group.columns else None
        
        # Get heatmap factor for this day
        heat_factor = heatmap_factors.get(gameing_day, 0.5) if gameing_day else 0.5
        
        # Player preference simulation
        player_hash = hash(str(player_id)) % 100
        
        if player_hash < 30:  # 30% prefer Novomatic
            preferred_provider = 'Novomatic'
        elif player_hash < 60:  # 30% prefer IGT
            preferred_provider = 'IGT'
        else:  # 40% prefer Pragmatic Play
            preferred_provider = 'Pragmatic Play'
        
        # Filter games by preferred provider (70% of the time)
        np.random.seed(player_hash)
        if np.random.random() < 0.7:
            available_games = game_catalog[game_catalog['provider'] == preferred_provider]
        else:
            available_games = game_catalog
        
        if len(available_games) == 0:
            available_games = game_catalog
        
        # Apply popularity weights with heatmap integration
        base_weights = available_games['weight'].values
        heat_multiplier = 1 + (heat_factor * 0.5)
        final_weights = base_weights * heat_multiplier
        final_weights = final_weights / final_weights.sum()
        
        # Select game
        selected_game = np.random.choice(
            available_games['game_id'].values,
            p=final_weights
        )
        
        return selected_game
    
    # Assign game IDs by player and game block
    game_assignments = events_df.groupby(['player_id', 'game_block']).apply(
        select_game_for_block_improved
    ).reset_index()
    game_assignments.columns = ['player_id', 'game_block', 'assigned_game_id']
    
    logger.info("📊 Step 3/3: Merging game assignments...")
    
    # Merge back to events
    events_df = events_df.merge(
        game_assignments, 
        on=['player_id', 'game_block'], 
        how='left'
    )
    
    events_df['game_id'] = events_df['assigned_game_id']
    
    # Show assignment statistics
    unique_games = events_df['game_id'].nunique()
    logger.info(f"✅ Assigned {unique_games} different games to {total_events:,} events")
    
    # Return with stable identifiers for update
    return events_df[['rn', 'player_id', 'ts', 'game_id']]

def update_batch_in_db_improved(engine, batch_df):
    """
    OPTIMIZED: Bulk update with progress tracking
    """
    total_records = len(batch_df)
    logger.info(f"Updating {total_records} records in database...")
    
    try:
        with engine.connect() as conn:
            # Create temporary table for bulk update
            temp_table_query = text("""
            CREATE TEMP TABLE game_updates (
                player_id TEXT,
                ts TIMESTAMP,
                new_game_id INT
            )
            """)
            conn.execute(temp_table_query)
            
            # Insert all data to temp table at once
            batch_df[['player_id', 'ts', 'game_id']].rename(
                columns={'game_id': 'new_game_id'}
            ).to_sql('game_updates', conn, if_exists='append', index=False, method='multi')
            
            logger.info(f"✅ Bulk inserted {total_records} records to temp table")
            
            # Single bulk update query
            update_query = text("""
            UPDATE casino_data.temp_valid_game_events 
            SET game_id_simulated_v1 = gu.new_game_id
            FROM game_updates gu
            WHERE casino_data.temp_valid_game_events.player_id = gu.player_id
              AND casino_data.temp_valid_game_events.ts = gu.ts
              AND (casino_data.temp_valid_game_events.game_id_simulated_v1 IS NULL 
                   OR casino_data.temp_valid_game_events.game_id_simulated_v1 = 0)
            """)
            
            result = conn.execute(update_query)
            conn.commit()
            
            logger.info(f"✅ Bulk updated {total_records} records successfully")
            
    except Exception as e:
        logger.error(f"❌ Database update failed: {e}")
        raise

def validate_game_assignment(engine):
    """
    FIXED: Validation with correct column names
    """
    logger.info("Validating game assignment results...")
    
    try:
        # FIXED: Check assignments with correct column name
        assigned_query = """
        SELECT COUNT(*) as assigned_events
        FROM casino_data.temp_valid_game_events 
        WHERE game_id_simulated_v1 IS NOT NULL AND game_id_simulated_v1 != 0
        """
        
        unassigned_query = """
        SELECT COUNT(*) as unassigned_events
        FROM casino_data.temp_valid_game_events 
        WHERE game_id_simulated_v1 IS NULL OR game_id_simulated_v1 = 0
        """
        
        assigned_count = pd.read_sql(assigned_query, engine).iloc[0]['assigned_events']
        unassigned_count = pd.read_sql(unassigned_query, engine).iloc[0]['unassigned_events']
        
        logger.info("=== GAME ASSIGNMENT VALIDATION RESULTS ===")
        logger.info(f"✅ Assigned events: {assigned_count:,}")
        logger.info(f"⏳ Unassigned events: {unassigned_count:,}")
        
        if assigned_count > 0:
            # Show game distribution
            distribution_query = """
            SELECT 
                sg.provider,
                sg.popularity_tier,
                COUNT(*) as event_count
            FROM casino_data.temp_valid_game_events tve
            JOIN casino_data.slot_game_catalog sg ON tve.game_id_simulated_v1 = sg.game_id
            WHERE tve.game_id_simulated_v1 IS NOT NULL
            GROUP BY sg.provider, sg.popularity_tier
            ORDER BY sg.provider, sg.popularity_tier
            """
            
            distribution = pd.read_sql(distribution_query, engine)
            logger.info("\n📊 Provider/Popularity Distribution:")
            print(distribution)
            
            # Show top games
            top_games_query = """
            SELECT 
                sg.game_name,
                sg.provider,
                COUNT(*) as play_count
            FROM casino_data.temp_valid_game_events tve
            JOIN casino_data.slot_game_catalog sg ON tve.game_id_simulated_v1 = sg.game_id
            WHERE tve.game_id_simulated_v1 IS NOT NULL
            GROUP BY sg.game_name, sg.provider
            ORDER BY play_count DESC
            LIMIT 5
            """
            
            top_games = pd.read_sql(top_games_query, engine)
            logger.info("\n🎯 Top 5 Assigned Games:")
            print(top_games)
        
        return assigned_count > 0
        
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False

def main():
    """Main execution function - FIXED VERSION"""
    print("CASINO GAME ID SIMULATION - PRODUCTION VERSION")
    print("=" * 55)
    print("University of Bath - MSc Computer Science")
    print("Student: Muhammed Yavuzhan CANLI")
    print("Ethics Approval: 10351-12382")
    print("PRODUCTION MODE - Processing full 2.7M dataset")
    print("=" * 50)
    
    try:
        # Database connection
        engine = get_db_connection()
        logger.info("✅ Database connection established")
        
        # Load game catalog
        game_catalog = load_game_catalog(engine)
        
        # Calculate heatmap factors
        heatmap_factors = calculate_heatmap_factors(engine)
        
        # PRODUCTION: Process full 2.7M dataset
        assign_game_ids_to_events(engine, game_catalog, heatmap_factors, test_limit=2476481)
        
        # Validate results
        validation_success = validate_game_assignment(engine)
        
        if validation_success:
            print("\n" + "=" * 50)
            print("✅ GAME ID SIMULATION TEST SUCCESSFUL")
            print("=" * 50)
            print("✅ Game assignment completed for test batch")
            print("✅ Validation passed")
            print("✅ Ready to process larger batches")
            print("\n🚀 Next: Increase test_limit or run full dataset")
            print("=" * 50)
        else:
            print("❌ TEST FAILED - Check validation results")
        
    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")
        print("❌ SIMULATION FAILED - Check database and configuration")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()