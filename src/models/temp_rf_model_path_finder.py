"""
Model Path Finder - Locate and verify existing model files
Step 1: Find available model files before running ensemble
"""

import os
import glob
from pathlib import Path

def find_model_files():
    """Find all .pkl model files in the project directory."""
    
    print("🔍 SEARCHING FOR MODEL FILES...")
    print("=" * 50)
    
    # Current directory and subdirectories
    search_patterns = [
        "*.pkl",
        "models/*.pkl", 
        "models/**/*.pkl",
        "**/*.pkl"
    ]
    
    found_models = {}
    
    for pattern in search_patterns:
        files = glob.glob(pattern, recursive=True)
        for file_path in files:
            if 'rf' in file_path.lower() and '2022' in file_path or '2023' in file_path:
                # Extract period from filename
                filename = os.path.basename(file_path)
                if '2022-H1' in filename:
                    found_models['2022-H1'] = file_path
                elif '2022-H2' in filename:
                    found_models['2022-H2'] = file_path
                elif '2023-H1' in filename:
                    found_models['2023-H1'] = file_path
                elif '2023-H2' in filename:
                    found_models['2023-H2'] = file_path
    
    print("✅ FOUND MODEL FILES:")
    if found_models:
        for period, path in found_models.items():
            file_size = os.path.getsize(path) / (1024*1024)  # MB
            print(f"  {period}: {path} ({file_size:.1f} MB)")
    else:
        print("  ❌ No model files found!")
        
    print("\n📁 DIRECTORY STRUCTURE:")
    for root, dirs, files in os.walk('.'):
        level = root.replace('.', '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            if file.endswith('.pkl'):
                print(f"{subindent}{file}")
    
    return found_models

def generate_corrected_paths(found_models):
    """Generate corrected model paths for ensemble script."""
    
    if not found_models:
        print("\n❌ NO MODELS FOUND - Cannot generate ensemble config")
        return None
        
    print("\n📝 CORRECTED MODEL PATHS FOR ENSEMBLE:")
    print("Copy this into your ensemble script:")
    print("-" * 40)
    
    config_lines = [
        "model_paths = {"
    ]
    
    for period in ['2022-H1', '2022-H2', '2023-H1', '2023-H2']:
        if period in found_models:
            # Convert to forward slashes for cross-platform compatibility
            path = found_models[period].replace('\\', '/')
            config_lines.append(f"    '{period}': '{path}',")
        else:
            config_lines.append(f"    # '{period}': 'PATH_NOT_FOUND',")
    
    config_lines.append("}")
    
    print("\n".join(config_lines))
    
    return found_models

if __name__ == "__main__":
    found_models = find_model_files()
    generate_corrected_paths(found_models)
    
    print(f"\n🎯 NEXT STEPS:")
    if found_models:
        print("1. Copy the corrected model_paths dictionary above")
        print("2. Update your ensemble script with these paths") 
        print("3. Run the ensemble script again")
    else:
        print("1. First train your models using clean_harmonized_rf_training.py")
        print("2. Verify model files are saved in models/ directory")
        print("3. Then run this script again to find the paths")