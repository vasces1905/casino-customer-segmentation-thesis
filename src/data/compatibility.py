# src/data/compatibility_layer.py

"""
Synthetic to Real Data Compatibility Layer
=========================================
Original work by: Muhammed Yavuzhan CANLI
University of Bath - MSc Computer Science
Ethics Approval: 10351-12382
Academic use only - No commercial distribution

This module bridges Casino-1 (synthetic CSV) and Casino-2 (real DB) data formats.
"""

from typing import Dict, List, Optional
import pandas as pd


class SyntheticToRealMapper:
    """
    Maps features between synthetic proof-of-concept (casino-1) and real casino data (casino-2).
    Ensures thesis continuity and reproducibility.
    """
    
    # Feature mapping from Casino-1 to Casino-2
    FEATURE_MAPPING = {
        # Casino-1 (CSV)        ---->  Casino-2 (PostgreSQL query)
        'customer_id':          'customer_id',  # Direct mapping
        'avg_bet':              """
            SELECT AVG(bet_amount) 
            FROM player_sessions 
            WHERE customer_id = %s
        """,
        'session_count':        """
            SELECT COUNT(DISTINCT session_id) 
            FROM player_sessions 
            WHERE customer_id = %s
        """,
        'total_wagered':        """
            SELECT SUM(total_bet) 
            FROM player_sessions 
            WHERE customer_id = %s
        """,
        'avg_session_duration': """
            SELECT AVG(EXTRACT(EPOCH FROM (end_time - start_time))/60) 
            FROM player_sessions 
            WHERE customer_id = %s AND end_time IS NOT NULL
        """,
        'favorite_game_type':   """
            SELECT game_type 
            FROM (
                SELECT game_type, COUNT(*) as play_count
                FROM player_sessions
                WHERE customer_id = %s
                GROUP BY game_type
                ORDER BY play_count DESC
                LIMIT 1
            ) t
        """,
        'zone_diversity':       """
            SELECT COUNT(DISTINCT machine_zone) 
            FROM player_sessions ps
            JOIN casino_machines cm ON ps.machine_id = cm.machine_id
            WHERE ps.customer_id = %s
        """,
        'loss_percentage':      """
            SELECT 
                CASE 
                    WHEN SUM(total_bet) > 0 
                    THEN (SUM(total_bet) - SUM(total_win)) / SUM(total_bet) * 100
                    ELSE 0 
                END
            FROM player_sessions
            WHERE customer_id = %s
        """,
        'days_since_last_visit': """
            SELECT EXTRACT(DAY FROM (CURRENT_DATE - MAX(DATE(start_time))))
            FROM player_sessions
            WHERE customer_id = %s
        """
    }
    
    # Segment mapping
    SEGMENT_MAPPING = {
        # Casino-1 segments → Casino-2 interpretation
        0: "Casual_Player",
        1: "High_Roller", 
        2: "Regular_Visitor",
        3: "At_Risk_Player"
    }
    
    def get_feature_query(self, feature_name: str) -> Optional[str]:
        """Get PostgreSQL query for a Casino-1 feature"""
        return self.FEATURE_MAPPING.get(feature_name)
    
    def map_segment_name(self, segment_id: int) -> str:
        """Convert numeric segment to business-friendly name"""
        return self.SEGMENT_MAPPING.get(segment_id, f"Segment_{segment_id}")
    
    def validate_mapping(self, synthetic_df: pd.DataFrame, 
                        real_df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate that all synthetic features can be mapped to real data.
        Used for thesis validation chapter.
        """
        validation_results = {}
        
        for col in synthetic_df.columns:
            if col in self.FEATURE_MAPPING:
                # Check if we can calculate this feature from real data
                validation_results[col] = True
            else:
                validation_results[col] = False
                
        return validation_results