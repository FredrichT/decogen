"""
Setup script to create the necessary directory structure for the project.
Run this script before starting to ensure all directories are created.
"""
import os
import sys
from pathlib import Path


def create_directory_structure():
    """Create the necessary directory structure for the project."""
    # Define directories to create
    directories = [
        'data',
        'data/raw',
        'data/cyclegan_data',
        'data/cyclegan_data/trainA',
        'data/cyclegan_data/trainB',
        'data/cyclegan_data/testA',
        'data/cyclegan_data/testB',
        'checkpoints',
        'samples',
        'evaluation_results',
        'inference_results',
        'models',
    ]
    
    # Create each directory
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    # Create __init__.py in models directory if it doesn't exist
    models_init_path = os.path.join('models', '__init__.py')
    if not os.path.exists(models_init_path):
        with open(models_init_path, 'w') as f:
            f.write('"""Models package for Interior Design Style Transfer project."""\n')
        print(f"Created file: {models_init_path}")
    
    print("\nDirectory structure created successfully!")
    print("You can now run the project with: python main.py [command]")


def check_files():
    """Check if main project files exist."""
    required_files = [
        'config.py',
        'data_preparation.py',
        'evaluate.py',
        'inference.py',
        'main.py',
        'train.py',
        'utils.py',
        'models/cycle_gan.py',
        'models/losses.py',
        'models/networks.py',
        'requirements.txt',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("\nWARNING: The following required files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nPlease ensure all required files are present before running the project.")
    else:
        print("\nAll required files are present.")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" " * 20 + "SETTING UP PROJECT STRUCTURE")
    print(" " * 10 + "Interior Design Style Transfer using CycleGAN")
    print("=" * 80 + "\n")
    
    # Create directory structure
    create_directory_structure()
    
    # Check for required files
    check_files()
    
    print("\nSetup complete!")