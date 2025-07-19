# test_hybrid_feature_engineering.py

"""
Test Implementation of Hybrid Feature Engineering
================================================
University of Bath - MSc Computer Science
Student: Muhammed Yavuzhan CANLI
Ethics Approval: 10351-12382

This test script demonstrates the complete hybrid feature engineering process
for the casino customer segmentation thesis project.

PURPOSE: Test and validate the hybrid approach that combines existing
processed data with novel academic contributions.
"""

import sys
import os
import logging
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import our educational hybrid feature engineer
from src.features.educational_hybrid_feature_engineering import EducationalHybridFeatureEngineer
from src.data.db_connector import AcademicDBConnector

# Setup logging for educational demonstration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_feature_engineering.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def test_database_connection():
    """
    Test database connection and verify data availability.
    
    EDUCATIONAL PURPOSE:
    This function demonstrates proper database connectivity testing
    and data availability verification for academic projects.
    """
    print("STEP 0: Testing Database Connection")
    print("=" * 50)
    
    try:
        # Initialize database connector
        db_connector = AcademicDBConnector()
        
        # Test connection
        with db_connector.get_connection() as conn:
            with conn.cursor() as cursor:
                # Test basic connectivity
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                print(f"Database connected successfully")
                print(f"PostgreSQL version: {version[0]}")
                
                # Check key tables
                key_tables = [
                    'customer_behavior_profiles',
                    'customer_game_preferences', 
                    'temp_valid_game_events'
                ]
                
                for table_name in key_tables:
                    cursor.execute(f"""
                        SELECT COUNT(*) 
                        FROM casino_data.{table_name}
                    """)
                    count = cursor.fetchone()[0]
                    print(f"{table_name}: {count:,} records")
        
        print("Database connection test completed successfully")
        return True
        
    except Exception as error:
        print(f"Database connection failed: {error}")
        return False


def run_hybrid_feature_engineering_test():
    """
    Execute the complete hybrid feature engineering process.
    
    EDUCATIONAL WORKFLOW:
    This function demonstrates the complete academic workflow for
    hybrid feature engineering in casino customer analytics.
    """
    print("\n🎓 UNIVERSITY OF BATH - HYBRID FEATURE ENGINEERING TEST")
    print("=" * 70)
    print("Student: Muhammed Yavuzhan CANLI")
    print("Ethics Approval: 10351-12382")
    print("Objective: Test hybrid feature engineering for thesis")
    print("=" * 70)
    
    try:
        # Step 1: Test database connection
        connection_success = test_database_connection()
        if not connection_success:
            print("Cannot proceed without database connection")
            return False
        
        # Step 2: Initialize educational feature engineer
        print("\nSTEP 1: Initializing Educational Feature Engineer")
        print("=" * 60)
        
        db_connector = AcademicDBConnector()
        feature_engineer = EducationalHybridFeatureEngineer(db_connector)
        
        print("Educational Hybrid Feature Engineer initialized")
        print(f"Academic Ethics Reference: {feature_engineer.academic_metadata['ethics_approval_reference']}")
        print(f"Institution: {feature_engineer.academic_metadata['institution']}")
        
        # Step 3: Execute complete feature engineering process
        print("\n STEP 2: Executing Complete Hybrid Feature Engineering")
        print("=" * 65)
        
        logger.info("Starting hybrid feature engineering test execution")
        
        # Execute the complete workflow
        complete_feature_matrix = feature_engineer.execute_complete_hybrid_feature_engineering()
        
        # Step 4: Validate results
        print("\n STEP 3: Validating Feature Engineering Results")
        print("=" * 55)
        
        if complete_feature_matrix is not None:
            print("Feature engineering completed successfully!")
            print(f"Feature matrix shape: {complete_feature_matrix.shape}")
            print(f"Customers processed: {len(complete_feature_matrix):,}")
            print(f"Features per customer: {len(complete_feature_matrix.columns)}")
            
            # Display sample of created features
            print("\n Sample of Created Features:")
            feature_columns = list(complete_feature_matrix.columns)
            sample_features = feature_columns[:15]  # Show first 15 features
            
            for i, feature_name in enumerate(sample_features, 1):
                print(f"  {i:2d}. {feature_name}")
            
            if len(feature_columns) > 15:
                print(f"  ... and {len(feature_columns) - 15} more features")
            
            # Show coverage statistics
            print("\n Coverage Statistics:")
            total_expected_customers = 38319  # From our database analysis
            actual_customers = len(complete_feature_matrix)
            coverage_percentage = (actual_customers / total_expected_customers) * 100
            
            print(f"Target customers: {total_expected_customers:,}")
            print(f"Processed customers: {actual_customers:,}")
            print(f"Coverage achieved: {coverage_percentage:.1f}%")
            
            if coverage_percentage >= 95:
                print("EXCELLENT: 95%+ coverage achieved!")
            elif coverage_percentage >= 85:
                print("GOOD: 85%+ coverage achieved!")
            else:
                print("Coverage below target, investigation needed")
            
            # Check for academic features
            academic_features = [
                'academic_loss_chasing_score',
                'weekend_preference_ratio',
                'late_night_player_ratio',
                'betting_volatility_coefficient'
            ]
            
            print("\n🔬 Academic Feature Validation:")
            for feature in academic_features:
                if feature in complete_feature_matrix.columns:
                    print(f" {feature}: Present")
                else:
                    print(f" {feature}: Missing")
            
            # Display sample data for verification
            print("\n Sample Customer Data (First 3 customers):")
            sample_data = complete_feature_matrix.head(3)
            key_display_columns = [
                'customer_id', 'overall_avg_bet', 'total_events', 
                'loss_rate_percentage', 'academic_loss_chasing_score'
            ]
            
            available_display_columns = [col for col in key_display_columns if col in sample_data.columns]
            
            if available_display_columns:
                print(sample_data[available_display_columns].to_string())
            else:
                print("Sample data columns not available for display")
            
            return True
            
        else:
            print(" Problem - Feature engineering failed - no feature matrix returned")
            return False
            
    except Exception as error:
        logger.error(f"Hybrid feature engineering test failed: {error}")
        print(f" Problem - Test execution failed: {error}")
        return False


def validate_ml_readiness():
    """
    Validate that the feature matrix is ready for ML model consumption.
    
    EDUCATIONAL PURPOSE:
    This function demonstrates how to validate feature matrices for
    machine learning applications in academic research.
    """
    print("\n STEP 4: ML Readiness Validation")
    print("=" * 40)
    
    try:
        db_connector = AcademicDBConnector()
        
        # Check if features exist in database
        with db_connector.get_connection() as conn:
            # Check customer_features table
            validation_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT customer_id) as unique_customers,
                COUNT(CASE WHEN overall_avg_bet IS NOT NULL THEN 1 END) as valid_avg_bet,
                COUNT(CASE WHEN academic_loss_chasing_score IS NOT NULL THEN 1 END) as valid_academic_features
            FROM casino_data.customer_features
            """
            
            import pandas as pd
            validation_results = pd.read_sql_query(validation_query, conn)
            
            if len(validation_results) > 0:
                total_records = validation_results['total_records'].iloc[0]
                unique_customers = validation_results['unique_customers'].iloc[0]
                valid_avg_bet = validation_results['valid_avg_bet'].iloc[0]
                valid_academic_features = validation_results['valid_academic_features'].iloc[0]
                
                print(f"Total feature records: {total_records:,}")
                print(f"Unique customers: {unique_customers:,}")
                print(f"Valid avg_bet records: {valid_avg_bet:,}")
                print(f"Valid academic features: {valid_academic_features:,}")
                
                # Calculate readiness scores
                basic_readiness = (valid_avg_bet / max(1, total_records)) * 100
                academic_readiness = (valid_academic_features / max(1, total_records)) * 100
                
                print(f"\n ML Readiness Assessment:")
                print(f"  Basic features readiness: {basic_readiness:.1f}%")
                print(f"  Academic features readiness: {academic_readiness:.1f}%")
                
                if basic_readiness >= 95 and academic_readiness >= 95:
                    print("READY FOR ML PIPELINE!")
                    print("K-means clustering ready")
                    print("Random Forest training ready")
                    print("CRM integration ready")
                    return True
                else:
                    print("ML readiness incomplete")
                    return False
            else:
                print("Problem - No feature data found in database")
                return False
                
    except Exception as error:
        print(f"ML readiness validation failed: {error}")
        return False


def main():
    """
    Main test execution function.
    
    EDUCATIONAL PURPOSE:
    This main function demonstrates the complete test workflow for
    academic feature engineering validation.
    """
    start_time = datetime.now()
    
    print("🎓 HYBRID FEATURE ENGINEERING TEST EXECUTION")
    print("=" * 50)
    print(f"Test started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Execute hybrid feature engineering test
    feature_engineering_success = run_hybrid_feature_engineering_test()
    
    if feature_engineering_success:
        # Validate ML readiness
        ml_ready = validate_ml_readiness()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n Test completed at: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total duration: {duration}")
        
        if ml_ready:
            print("\n SUCCESS: HYBRID FEATURE ENGINEERING COMPLETED!")
            print("System ready for next phase: K-means Customer Segmentation")
            print("Next step: python run_customer_segmentation.py")
        else:
            print("\n Feature engineering completed but ML readiness validation failed")
            print("Recommendation: Check feature quality and completeness")
    else:
        print("\n FEATURE ENGINEERING TEST FAILED")
        print(" Please check logs for detailed error information")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()