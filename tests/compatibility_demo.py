from src.data.compatibility_demo import SyntheticToRealMapper
import pandas as pd

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

    # Example 3: Validate synthetic to real mapping
    synthetic_df = pd.DataFrame(columns=["customer_id", "avg_bet", "zone_diversity", "unknown_feature"])
    validation_result = mapper.validate_mapping(synthetic_df, pd.DataFrame())
    print("Validation result:")
    for feature, is_mapped in validation_result.items():
        print(f"  {feature}: {'Mapped' if is_mapped else 'No mapping found'}")

if __name__ == "__main__":
    demo_mapper()