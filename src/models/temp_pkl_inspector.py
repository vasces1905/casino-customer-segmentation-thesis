"""
REAL PKL Inspector - Using CORRECT paths this time!
Let's find out which PKLs are actually good quality
"""

import joblib
import numpy as np
import pandas as pd

def inspect_pkl_detailed(path, period, model_type):
    """Detailed inspection of PKL file quality."""
    print(f"\n🔍 INSPECTING {model_type} - {period}")
    print(f"📁 Path: {path}")
    print("-" * 60)
    
    try:
        data = joblib.load(path)
        
        print(f"📋 PKL Contents:")
        for key, value in data.items():
            print(f"   {key}: {type(value)}")
        
        # Get the RF model
        rf_model = data.get('rf_model') or data.get('model')
        scaler = data.get('scaler')
        label_encoder = data.get('label_encoder')
        feature_names = data.get('feature_names') or data.get('features')
        performance = data.get('performance', {})
        
        # QUALITY CHECKS
        quality_score = 0
        issues = []
        
        # Check 1: RF Model exists and is trained
        if rf_model and hasattr(rf_model, 'classes_'):
            classes = rf_model.classes_
            n_features = getattr(rf_model, 'n_features_in_', 'Unknown')
            print(f"✅ RandomForest: {len(classes)} classes {classes}, {n_features} features")
            quality_score += 3
        else:
            print(f"❌ RandomForest: Missing or not trained")
            issues.append("No valid RF model")
        
        # Check 2: Scaler exists
        if scaler and hasattr(scaler, 'scale_'):
            print(f"✅ StandardScaler: Available ({len(scaler.scale_)} features)")
            quality_score += 2
        else:
            print(f"❌ StandardScaler: Missing")
            issues.append("No scaler")
        
        # Check 3: Label encoder exists
        if label_encoder and hasattr(label_encoder, 'classes_'):
            print(f"✅ LabelEncoder: {len(label_encoder.classes_)} classes {label_encoder.classes_}")
            quality_score += 2
        else:
            print(f"❌ LabelEncoder: Missing")
            issues.append("No label encoder")
        
        # Check 4: Feature names available
        if feature_names and len(feature_names) > 0:
            print(f"✅ Features: {len(feature_names)} features")
            print(f"   First 5: {feature_names[:5]}")
            quality_score += 2
        else:
            print(f"❌ Features: Missing feature names")
            issues.append("No feature names")
        
        # Check 5: Performance metrics
        if performance and len(performance) > 0:
            print(f"✅ Performance: {list(performance.keys())}")
            for perf_key, perf_val in performance.items():
                if isinstance(perf_val, (int, float)):
                    print(f"   {perf_key}: {perf_val:.3f}")
                else:
                    print(f"   {perf_key}: {perf_val}")
            quality_score += 1
        else:
            print(f"⚠️ Performance: No metrics stored")
        
        # FINAL QUALITY ASSESSMENT
        max_score = 10
        quality_percentage = (quality_score / max_score) * 100
        
        print(f"\n📊 QUALITY ASSESSMENT:")
        print(f"   Score: {quality_score}/{max_score} ({quality_percentage:.1f}%)")
        
        if quality_percentage >= 80:
            print(f"   🏆 EXCELLENT - Ready for production")
        elif quality_percentage >= 60:
            print(f"   ✅ GOOD - Usable with minor issues")
        elif quality_percentage >= 40:
            print(f"   ⚠️ MODERATE - Has significant issues")
        else:
            print(f"   ❌ POOR - Not recommended")
        
        if issues:
            print(f"   Issues: {', '.join(issues)}")
        
        return {
            'path': path,
            'period': period,
            'model_type': model_type,
            'quality_score': quality_score,
            'quality_percentage': quality_percentage,
            'classes': classes if rf_model and hasattr(rf_model, 'classes_') else [],
            'n_features': len(feature_names) if feature_names else 0,
            'has_scaler': scaler is not None,
            'has_label_encoder': label_encoder is not None,
            'performance_metrics': performance,
            'issues': issues,
            'data': data  # Store for later use
        }
        
    except Exception as e:
        print(f"❌ FAILED to load: {e}")
        return {
            'path': path,
            'period': period,
            'model_type': model_type,
            'quality_score': 0,
            'quality_percentage': 0,
            'issues': [f"Load failed: {e}"],
            'data': None
        }

def compare_model_types():
    """Compare BALANCED vs GENERIC_RF models."""
    
    print("🔍 COMPARING MODEL TYPES - BALANCED vs GENERIC_RF")
    print("=" * 70)
    
    # Define paths for comparison
    models_to_compare = [
        # BALANCED models
        ('models/balanced/balanced_rf_2022-H1_20250728_1016.pkl', '2022-H1', 'BALANCED'),
        ('models/balanced/balanced_rf_2022-H2_20250728_1017.pkl', '2022-H2', 'BALANCED'),
        ('models/balanced/balanced_rf_2023-H1_20250728_1018.pkl', '2023-H1', 'BALANCED'),
        ('models/balanced/balanced_rf_2023-H2_20250728_1020.pkl', '2023-H2', 'BALANCED'),
        
        # GENERIC_RF models
        ('models/generic_rf/clean_harmonized_rf_2022-H1_v1_20250728_1145.pkl', '2022-H1', 'GENERIC_RF'),
        ('models/generic_rf/clean_harmonized_rf_2022-H2_v1_20250728_1201.pkl', '2022-H2', 'GENERIC_RF'),
        ('models/generic_rf/clean_harmonized_rf_2023-H1_v1_20250728_1204.pkl', '2023-H1', 'GENERIC_RF'),
        ('models/generic_rf/clean_harmonized_rf_2023-H2_v1_20250728_1206.pkl', '2023-H2', 'GENERIC_RF'),
    ]
    
    results = []
    
    for path, period, model_type in models_to_compare:
        result = inspect_pkl_detailed(path, period, model_type)
        results.append(result)
    
    # Summary comparison
    print(f"\n" + "="*70)
    print(f"📊 SUMMARY COMPARISON")
    print("="*70)
    
    df_results = pd.DataFrame([
        {
            'Period': r['period'],
            'Type': r['model_type'],
            'Quality%': r['quality_percentage'],
            'Classes': len(r['classes']),
            'Features': r['n_features'],
            'Issues': len(r['issues'])
        }
        for r in results if r['data'] is not None
    ])
    
    if not df_results.empty:
        print(f"\n📋 RESULTS TABLE:")
        print(df_results.to_string(index=False))
        
        # Find best performing models
        print(f"\n🏆 BEST MODELS BY PERIOD:")
        for period in ['2022-H1', '2022-H2', '2023-H1', '2023-H2']:
            period_results = [r for r in results if r['period'] == period and r['data'] is not None]
            if period_results:
                best = max(period_results, key=lambda x: x['quality_percentage'])
                print(f"   {period}: {best['model_type']} ({best['quality_percentage']:.1f}%)")
        
        # Overall recommendation
        balanced_avg = df_results[df_results['Type'] == 'BALANCED']['Quality%'].mean()
        generic_avg = df_results[df_results['Type'] == 'GENERIC_RF']['Quality%'].mean()
        
        print(f"\n🎯 OVERALL RECOMMENDATION:")
        print(f"   BALANCED average quality: {balanced_avg:.1f}%")
        print(f"   GENERIC_RF average quality: {generic_avg:.1f}%")
        
        if balanced_avg > generic_avg:
            print(f"   🏆 RECOMMENDATION: Use BALANCED models")
            recommended_type = 'BALANCED'
        else:
            print(f"   🏆 RECOMMENDATION: Use GENERIC_RF models")
            recommended_type = 'GENERIC_RF'
        
        return results, recommended_type
    else:
        print("❌ No valid models found!")
        return results, None

def main():
    """Main inspection and comparison."""
    
    from datetime import datetime
    
    # Setup logging to file
    log_file = f"pkl_quality_inspection_{datetime.now().strftime('%Y%m%d_%H%M')}.log"
    
    import sys
    import io
    
    # Capture all print output
    class Logger:
        def __init__(self, filename):
            self.terminal = sys.stdout
            self.log = open(filename, "w", encoding='utf-8')

        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()

        def flush(self):
            self.terminal.flush()
            self.log.flush()
            
        def close(self):
            self.log.close()

    logger = Logger(log_file)
    sys.stdout = logger
    
    print("🔍 REAL PKL QUALITY INSPECTOR")
    print("=" * 50)
    print("Finding the BEST quality models for predictions...")
    print(f"📝 Logging to: {log_file}")
    
    try:
        results, recommended_type = compare_model_types()
        
        if recommended_type:
            print(f"\n✅ INSPECTION COMPLETE!")
            print(f"🎯 RECOMMENDED MODEL TYPE: {recommended_type}")
            print(f"\n🚀 READY TO PROCEED WITH {recommended_type} MODELS")
            
            # Close logger
            logger.close()
            sys.stdout = logger.terminal
            
            print(f"📝 Detailed log saved to: {log_file}")
            
            return results, recommended_type
        else:
            print(f"\n❌ INSPECTION FAILED - No usable models found")
            
            # Close logger
            logger.close()
            sys.stdout = logger.terminal
            
            print(f"📝 Error log saved to: {log_file}")
            return None, None
            
    except Exception as e:
        print(f"❌ INSPECTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        
        # Close logger
        logger.close()
        sys.stdout = logger.terminal
        
        print(f"📝 Error log saved to: {log_file}")
        return None, None

if __name__ == "__main__":
    results, recommendation = main()