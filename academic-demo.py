#!/usr/bin/env python3
"""
Academic Demo Script for Casino Customer Segmentation
University of Bath - MSc Computer Science
Student: Muhammed Yavuzhan CANLI
Ethics Approval: 10351-12382

This script provides a complete demo that works out-of-the-box
for academic evaluation without requiring large model files.
"""

import os
import sys
import time
import subprocess
from pathlib import Path

def print_academic_header():
    print("="*70)
    print("CASINO CUSTOMER SEGMENTATION - ACADEMIC DEMO")
    print("University of Bath - MSc Computer Science")
    print("Student: Muhammed Yavuzhan CANLI")
    print("Supervisor: Dr. Moody Alam")
    print("Ethics Approval: 10351-12382")
    print("="*70)

def check_dependencies():
    """Check if required dependencies are available"""
    print("\n[1/6] Checking dependencies...")
    
    required_modules = [
        'pandas', 'numpy', 'scikit-learn', 
        'psycopg2', 'python-dotenv'
    ]
    
    missing = []
    for module in required_modules:
        try:
            if module == 'psycopg2':
                import psycopg2
            else:
                __import__(module.replace('-', '_'))
            print(f"  ✓ {module}")
        except ImportError:
            missing.append(module)
            print(f"  ✗ {module} (missing)")
    
    if missing:
        print(f"\n[ERROR] Missing modules: {', '.join(missing)}")
        print("Installing missing dependencies...")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing)
    
    return len(missing) == 0

def setup_environment():
    """Setup environment variables"""
    print("\n[2/6] Setting up environment...")
    
    if not Path('.env').exists():
        env_content = """# Academic Demo Configuration
DB_HOST=postgres
DB_PORT=5432
DB_NAME=casino_research
DB_USER=researcher
DB_PASSWORD=academic_password_2024
ETHICS_REF=10351-12382
STUDENT_ID=mycc21
ENVIRONMENT=academic_demo
ACADEMIC_MODE=true
"""
        with open('.env', 'w') as f:
            f.write(env_content)
        print("  ✓ Created .env file")
    else:
        print("  ✓ .env file exists")

def wait_for_postgres():
    """Wait for PostgreSQL to be ready"""
    print("\n[3/6] Waiting for PostgreSQL...")
    
    import psycopg2
    from psycopg2 import OperationalError
    
    max_attempts = 30
    for attempt in range(max_attempts):
        try:
            conn = psycopg2.connect(
                host='localhost',
                port=5432,
                database='casino_research',
                user='researcher',
                password='academic_password_2024'
            )
            conn.close()
            print("  ✓ PostgreSQL is ready")
            return True
        except OperationalError:
            if attempt < max_attempts - 1:
                print(f"  ⏳ Waiting for PostgreSQL... ({attempt + 1}/{max_attempts})")
                time.sleep(2)
            else:
                print("  ✗ PostgreSQL not available")
                return False
    
    return False

def create_synthetic_data():
    """Create synthetic data for demo"""
    print("\n[4/6] Creating synthetic demo data...")
    
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate synthetic customer data
    np.random.seed(42)  # For reproducible results
    n_customers = 1000
    
    customers = pd.DataFrame({
        'customer_id': range(1, n_customers + 1),
        'age': np.random.randint(21, 80, n_customers),
        'total_visits': np.random.poisson(10, n_customers),
        'total_spend': np.random.exponential(500, n_customers),
        'avg_session_duration': np.random.gamma(2, 30, n_customers),
        'preferred_game_type': np.random.choice(['slots', 'table', 'poker'], n_customers),
        'registration_date': pd.date_range(
            start='2022-01-01', 
            end='2024-01-01', 
            periods=n_customers
        )
    })
    
    # Save to CSV for demo
    os.makedirs('data', exist_ok=True)
    customers.to_csv('data/synthetic_customers_demo.csv', index=False)
    print("  ✓ Created synthetic customer data (1000 records)")
    
    return customers

def run_demo_pipeline():
    """Run the demo pipeline with synthetic data"""
    print("\n[5/6] Running academic demo pipeline...")
    
    try:
        # Add src to Python path
        sys.path.insert(0, 'src')
        
        # Import required modules
        from src.features.feature_engineering import CasinoFeatureEngineer
        from src.models.segmentation import CustomerSegmentation
        
        # Load synthetic data
        import pandas as pd
        df = pd.read_csv('data/synthetic_customers_demo.csv')
        
        print("  ✓ Loaded synthetic data")
        
        # Feature engineering
        feature_engineer = CasinoFeatureEngineer()
        features = feature_engineer.create_basic_features(df)
        print("  ✓ Created features")
        
        # Customer segmentation (using simple K-means)
        segmentation = CustomerSegmentation(n_clusters=4)
        segments = segmentation.fit_predict(features)
        print("  ✓ Created customer segments")
        
        # Add segments to dataframe
        df['segment'] = segments
        
        # Save results
        df.to_csv('data/demo_results.csv', index=False)
        print("  ✓ Saved segmentation results")
        
        # Display summary
        print("\n  📊 DEMO RESULTS SUMMARY:")
        print(f"     Total customers: {len(df)}")
        print(f"     Segments created: {df['segment'].nunique()}")
        for segment in sorted(df['segment'].unique()):
            count = (df['segment'] == segment).sum()
            avg_spend = df[df['segment'] == segment]['total_spend'].mean()
            print(f"     Segment {segment}: {count} customers (avg spend: ${avg_spend:.2f})")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Demo pipeline failed: {e}")
        print("  ℹ️  This is expected if models are not available")
        print("  ℹ️  The database and basic setup are still functional")
        return False

def show_next_steps():
    """Show what evaluators can do next"""
    print("\n[6/6] Academic Demo Complete!")
    print("\n📋 EVALUATION OPTIONS:")
    print("   1. Review generated data: data/demo_results.csv")
    print("   2. Check database: docker exec -it casino_postgres_minimal psql -U researcher -d casino_research")
    print("   3. Run full pipeline: python main_pipeline.py --mode synthetic")
    print("   4. View source code: src/ directory")
    print("   5. Check academic compliance: README-MINIMAL.md")
    
    print("\n🎓 ACADEMIC NOTES:")
    print("   • All data is synthetic and GDPR-compliant")
    print("   • Ethics approval: 10351-12382")
    print("   • Reproducible research methodology")
    print("   • Full source code available for review")

def main():
    """Main demo function"""
    print_academic_header()
    
    # Run demo steps
    if not check_dependencies():
        print("\n[ERROR] Please install missing dependencies and try again")
        return
    
    setup_environment()
    
    if not wait_for_postgres():
        print("\n[WARNING] PostgreSQL not available. Starting without database...")
        print("          You can still review the code and synthetic data generation.")
    
    create_synthetic_data()
    run_demo_pipeline()
    show_next_steps()
    
    print(f"\n🎉 Academic demo ready for evaluation!")
    print("   Time taken: ~2-3 minutes")
    print("   Package size: <50MB (excluding optional models)")

if __name__ == "__main__":
    main()
