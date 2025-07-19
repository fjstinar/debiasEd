#!/usr/bin/env python3
"""
DebiasEd Fairness GUI - Standalone Executable
==============================================

A user-friendly GUI application for bias mitigation in educational machine learning.
This script can be run directly without Docker or complex setup.

Usage:
    python run_fairness_gui.py

Requirements:
    - Python 3.7+
    - Required packages (will be checked and reported)
"""
import sys
import os
cwd = os.getcwd()
sys.path.append(cwd)
import subprocess
import importlib
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

# Required packages for the GUI
REQUIRED_PACKAGES = {
    'tkinter': 'tkinter',  # Usually built-in
    'numpy': 'numpy', 
    'pandas': 'pandas',
    'sklearn': 'scikit-learn',
    'imblearn': 'imbalanced-learn',
    'pickle': 'pickle'  # Built-in
}


def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("❌ Error: Python 3.7 or higher is required.")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_package_installation():
    """Check if required packages are installed"""
    missing_packages = []
    for package, pip_name in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(package)
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append((package, pip_name))
            print(f"❌ {package} is missing")
    return missing_packages

def install_missing_packages(missing_packages):
    """Attempt to install missing packages"""
    if not missing_packages:
        return True
    print("\n📦 Missing packages detected. Attempting to install...")
    for _, pip_name in missing_packages:
        if pip_name in ['tkinter', 'pickle']:  # Skip built-in packages
            continue
            
        try:
            print(f"   Installing {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            print(f"   ✅ {pip_name} installed successfully")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {pip_name}")
            return False
    
    return True

def setup_project_path():
    """Add project directories to Python path"""
    project_root = Path.cwd()
    src_path = os.path.join(project_root, "src")
    gui_path = os.path.join(project_root, "gui")
    
    if os.path.exists(src_path):
        sys.path.insert(0, src_path)
        print(f"Added src path: {src_path}")
    else:
        print(f"Source path not found: {src_path}")
    
    if os.path.exists(gui_path):
        sys.path.insert(0, gui_path)
        print(f"Added gui path: {gui_path}")
    else:
        print(f"GUI path not found: {gui_path}")
    
    return project_root

def check_data_directory():
    """Check if data directory exists and has datasets"""
    project_root = Path.cwd()
    data_path = os.path.join(project_root, "data")
    
    if not os.path.exists(data_path):
        print(f"Data directory not found: {data_path}")
        print("   The GUI will still work, but no datasets will be available.")
        return False
    
    # Count available datasets
    datasets = []
    try:
        for folder in os.listdir(data_path):
            potential_data_file = os.path.join(data_path, folder, "data_dictionary.pkl")
            if os.path.isfile(potential_data_file):
                datasets.append(folder)
    except Exception as e:
        print(f"Error scanning datasets: {e}")
    
    if datasets:
        print(f"Found {len(datasets)} datasets: {', '.join(datasets)}")
    else:
        print("No datasets found. You may need to prepare data first.")
    
    return len(datasets) > 0

def run_gui():
    """Launch the fairness GUI"""
    try:
        print("\nLaunching DebiasEd Fairness GUI...")
        from gui.unfairness_mitigation_gui import DataLoaderApp
        root = tk.Tk()
        root.withdraw()  # Hide initially
        
        # Show welcome message
        welcome_msg = """
            Welcome to DebiasEd Fairness GUI!

            This tool helps you:
            • Load educational datasets
            • Train machine learning models
            • Evaluate bias and fairness
            • Compare different approaches

            Click OK to continue to the main interface.
        """
        messagebox.showinfo("Welcome to DebiasEd", welcome_msg.strip())
        root.deiconify()  # Show the main window
        _ = DataLoaderApp(root)
        root.mainloop()
        
    except ImportError as e:
        print(f"Error importing GUI module: {e}")
        print("   Make sure the gui/unfairness_mitigation_gui.py file exists.")
        return False
    except Exception as e:
        print(f"Error running GUI: {e}")
        return False
    
    return True

def show_help():
    help_text = """
        DebiasEd Fairness GUI - Help
        ============================

        This is a standalone application for bias mitigation in educational ML.

        QUICK START:
        1. Run: python run_fairness_gui.py
        2. The GUI will open showing available datasets
        3. Click on a dataset to load it
        4. Use "Train Decision Tree" to run a basic model
        5. View results including fairness metrics

        TROUBLESHOOTING:
        - If packages are missing, the script will try to install them
        - Make sure you have internet connection for package installation
        - Data files should be in: data/[dataset]/data_dictionary.pkl

        FEATURES:
        - Load multiple educational datasets (EEDI, OULAD, etc.)
        - Train basic decision tree models
        - View dataset features and statistics
        - Evaluate model performance and fairness

        For advanced features, see the full research framework in src/
    """
    print(help_text)

def main():
    print("=" * 60)
    print("DebiasEd: Fairness in Educational Machine Learning")
    print("=" * 60)
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['-h', '--help', 'help']:
            show_help()
            return
        elif sys.argv[1] == '--check':
            print("Running system check only...")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for usage information")
            return
    print("\nRunning system checks...")
    
    if not check_python_version():
        return
    
    # Setup project paths
    project_root = setup_project_path()
    # Check package installation
    missing = check_package_installation()
    if missing and not install_missing_packages(missing):
        print("\n❌ Could not install required packages.")
        print("   Please install manually:")
        for package, pip_name in missing:
            if pip_name not in ['tkinter', 'pickle']:
                print(f"     pip install {pip_name}")
        return
    
    # Check data availability
    has_data = check_data_directory()
    
    # Exit if only checking
    if len(sys.argv) > 1 and sys.argv[1] == '--check':
        print("\n✅ System check complete!")
        return
    
    # Show status summary
    print("\n" + "=" * 40)
    print("SYSTEM STATUS SUMMARY")
    print("=" * 40)
    print(f"Python: {sys.version.split()[0]}")
    print(f"Required packages: Available")
    print(f"{'Found' if has_data else 'Limited'} Datasets: {'Available' if has_data else 'Limited'}")
    print(f"Project root: {project_root}")
    
    if not has_data:
        print("\nNote: Limited datasets available. The GUI will still work,")
        print("   but you may need to prepare data files first.")
    
    # Launch GUI
    success = run_gui()
    
    if success:
        print("\n👋 Thanks for using DebiasEd!")
    else:
        print("\n❌ GUI failed to start. Check error messages above.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("   Please report this issue if it persists.") 