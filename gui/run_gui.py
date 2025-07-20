#!/usr/bin/env python3
"""
Simple startup script for the DebiasEd GUI

This script ensures all dependencies are available and launches the GUI application.
"""

import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are available"""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'sklearn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Main function to launch the GUI"""
    print("=" * 60)
    print("DebiasEd - Bias Mitigation GUI")
    print("=" * 60)
    print()
    
    # Check if we're in the right directory
    if not Path("gui/debiased_jadouille_gui.py").exists():
        print("Error: debiased_jadouille_gui.py not found in gui directory")
        print("Please run this script from the project root directory")
        return 1
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        return 1
    
    print("All dependencies found")
    print()
    
    # Launch the GUI
    print("Launching GUI application...")
    try:
        # Add the gui directory to Python path
        gui_path = Path(__file__).parent
        sys.path.insert(0, str(gui_path))
        
        import debiased_jadouille_gui
        debiased_jadouille_gui.main()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 