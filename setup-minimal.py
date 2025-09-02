#!/usr/bin/env python3
"""
Minimal Setup Script for Casino Customer Segmentation Project
University of Bath - Academic Thesis
Student: Muhammed Yavuzhan CANLI
Ethics Approval: 10351-12382
"""

import os
import sys
import subprocess
import urllib.request
import zipfile
from pathlib import Path

def print_banner():
    print("="*60)
    print("Casino Customer Segmentation - Minimal Setup")
    print("University of Bath - MSc Computer Science")
    print("Student: Muhammed Yavuzhan CANLI")
    print("Ethics Approval: 10351-12382")
    print("="*60)

def check_docker():
    """Check if Docker is installed and running"""
    try:
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] Docker found:", result.stdout.strip())
            return True
    except FileNotFoundError:
        pass
    
    print("[ERROR] Docker not found. Please install Docker Desktop.")
    return False

def check_docker_compose():
    """Check if Docker Compose is available"""
    try:
        result = subprocess.run(['docker', 'compose', 'version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] Docker Compose found:", result.stdout.strip())
            return True
    except FileNotFoundError:
        pass
    
    print("[ERROR] Docker Compose not found.")
    return False

def download_models():
    """Download pre-trained models (placeholder for external storage)"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    print("[INFO] Models will be downloaded separately due to size constraints")
    print("   Please download from: [TO BE PROVIDED]")
    
    # Create placeholder files
    placeholder_models = [
        "clean_harmonized_rf_2022-H1_v1.pkl",
        "clean_harmonized_rf_2022-H2_v1.pkl", 
        "clean_harmonized_rf_2023-H1_v1.pkl",
        "clean_harmonized_rf_2023-H2_v1.pkl"
    ]
    
    for model in placeholder_models:
        placeholder_path = models_dir / f"{model}.placeholder"
        with open(placeholder_path, 'w') as f:
            f.write(f"Placeholder for {model}\n")
            f.write("Download the actual model from the provided link.\n")

def setup_environment():
    """Setup basic environment"""
    if not Path(".env").exists():
        if Path(".env.example").exists():
            print("[INFO] Copying .env.example to .env")
            import shutil
            shutil.copy(".env.example", ".env")
        else:
            print("[INFO] Creating basic .env file")
            with open(".env", "w") as f:
                f.write("DATABASE_URL=postgresql://researcher:academic_password_2024@localhost:5432/casino_research\n")
                f.write("ACADEMIC_MODE=true\n")

def build_minimal_setup():
    """Build the minimal Docker setup"""
    print("[INFO] Building minimal Docker setup...")
    
    try:
        # Build the minimal image
        result = subprocess.run([
            'docker', 'compose', '-f', 'docker-compose.minimal.yml', 'build'
        ], check=True)
        
        print("[OK] Docker images built successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Build failed: {e}")
        return False

def main():
    print_banner()
    
    # Check prerequisites
    if not check_docker():
        sys.exit(1)
        
    if not check_docker_compose():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Download/setup models
    download_models()
    
    # Build Docker setup
    if build_minimal_setup():
        print("\n[SUCCESS] Minimal setup complete!")
        print("\nNext steps:")
        print("1. Download models from provided links to ./models/")
        print("2. Run: docker compose -f docker-compose.minimal.yml up")
        print("3. Access the application on localhost")
        
        print(f"\nPackage size reduced to minimal core (~{get_core_size()}MB)")
    else:
        print("\n[ERROR] Setup failed. Please check the errors above.")
        sys.exit(1)

def get_core_size():
    """Estimate core package size"""
    core_dirs = ['src', 'schema', 'scripts']
    total_size = 0
    
    for dir_name in core_dirs:
        if os.path.exists(dir_name):
            for root, dirs, files in os.walk(dir_name):
                for file in files:
                    if not file.endswith(('.pkl', '.csv', '.txt', '.log')):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path):
                            total_size += os.path.getsize(file_path)
    
    return round(total_size / (1024 * 1024), 1)

if __name__ == "__main__":
    main()
