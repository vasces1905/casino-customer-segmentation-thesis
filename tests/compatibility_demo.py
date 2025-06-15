import sys
import os
import pandas as pd

# Ensure src/ is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data.compatibility_layer import SyntheticToRealMapper # type: ignore

def demo_mapper():
    mapper = SyntheticToRealMapper()

    # Example 1: Get SQL query for avg_bet
    feature = "avg_bet"
    query = mapper.get_feature_query(feature)
    if query:
        print(f"SQL for {feature}:\n{query}")
    else:
        print(f"No SQL mapping found for {feature}")

    # Example 2: Map segment ID to business label
    segment_id = 1
    segment_name = mapper.map_segment_name(segment_id)
    print(f"Segment {segment_id} maps to: {segment_name}")

    # Example 3: Validate mapping
    synthetic_df = pd.DataFrame(columns=["customer_id", "avg_bet", "zone_diversity", "unknown_feature"])
    result = mapper.validate_mapping(synthetic_df, pd.DataFrame())
    print("Mapping validation result:", result)

if __name__ == "__main__":
    demo_mapper()
