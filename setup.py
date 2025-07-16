#!/usr/bin/env python3
"""
Setup script for GFAN project
Installs dependencies and verifies installation
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def install_package(package):
    """Install a package using pip"""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print(f"✅ Successfully installed {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ Failed to install {package}")
        return False

def check_imports():
    """Check if required packages can be imported"""
    packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'pandas': 'Pandas',
        'tqdm': 'TQDM'
    }
    
    failed_imports = []
    
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✅ {name} imported successfully")
        except ImportError:
            print(f"❌ Failed to import {name}")
            failed_imports.append(package)
    
    return failed_imports

def setup_kaggle_environment():
    """Setup for Kaggle environment"""
    print("Setting up for Kaggle environment...")
    
    # Kaggle-specific packages
    kaggle_packages = [
        'mne',
        'pyedflib',
        'torch-geometric'
    ]
    
    for package in kaggle_packages:
        install_package(package)

def setup_local_environment():
    """Setup for local environment"""
    print("Setting up for local environment...")
    
    # Install from requirements.txt
    if os.path.exists('requirements.txt'):
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
            print("✅ Successfully installed requirements")
        except subprocess.CalledProcessError:
            print("❌ Failed to install requirements")

def verify_gpu():
    """Verify GPU availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            print("⚠️  GPU not available, using CPU")
            return False
    except ImportError:
        print("❌ Cannot check GPU - PyTorch not installed")
        return False

def create_directories():
    """Create necessary directories"""
    directories = [
        'data',
        'results',
        'checkpoints',
        'plots'
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def show_config_options():
    """Show available configuration options"""
    print("\n" + "=" * 50)
    print("CONFIGURATION OPTIONS")
    print("=" * 50)
    
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        print("\nAvailable Model Configurations:")
        for name, conf in config['model_configs'].items():
            print(f"  {name}: {conf['description']}")
        
        print("\nAvailable Data Configurations:")
        for name, conf in config['data_configs'].items():
            print(f"  {name}: {conf['description']}")
    else:
        print("Config file not found!")

def main():
    """Main setup function"""
    print("=" * 60)
    print("GFAN PROJECT SETUP")
    print("=" * 60)
    
    # Detect environment
    if os.path.exists('/kaggle'):
        print("🔍 Detected Kaggle environment")
        setup_kaggle_environment()
    else:
        print("🔍 Detected local environment")
        setup_local_environment()
    
    print("\n" + "=" * 50)
    print("VERIFYING INSTALLATION")
    print("=" * 50)
    
    # Check imports
    failed_imports = check_imports()
    
    # Verify GPU
    gpu_available = verify_gpu()
    
    # Create directories
    print("\n" + "=" * 50)
    print("CREATING DIRECTORIES")
    print("=" * 50)
    create_directories()
    
    # Show configuration options
    show_config_options()
    
    print("\n" + "=" * 60)
    print("SETUP SUMMARY")
    print("=" * 60)
    
    if failed_imports:
        print(f"❌ Failed imports: {', '.join(failed_imports)}")
        print("Please install missing packages manually")
    else:
        print("✅ All packages imported successfully")
    
    if gpu_available:
        print("✅ GPU acceleration available")
    else:
        print("⚠️  Using CPU (training will be slower)")
    
    print("\n🚀 Next steps:")
    print("1. For Kaggle: Copy kaggle_gfan_notebook.py to a new notebook")
    print("2. For local: Run python run_example.py for a demonstration")
    print("3. Check KAGGLE_QUICKSTART.md for detailed instructions")
    
    return len(failed_imports) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
