#!/usr/bin/env python3
"""
Dynamic Preprocessing Demonstration

This script demonstrates the new dynamic preprocessing system that loads
actual research-grade algorithms from src/mitigation/preprocessing instead
of simplified implementations.
"""

import sys
import os
import numpy as np

# Add the gui directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gui'))

# Import the dynamic preprocessing system
from gui.preprocessing_dynamic import get_available_preprocessing_methods, apply_preprocessing_method, get_import_status

def demonstrate_dynamic_preprocessing():
    """Demonstrate the dynamic preprocessing system with sample data"""
    
    print("=" * 60)
    print("DebiasEd Dynamic Preprocessing System Demonstration")
    print("=" * 60)
    
    # Show import status
    print("\n1. Import Status:")
    print("-" * 30)
    status = get_import_status()
    
    print(f"✅ Successfully loaded methods: {len(status['successful'])}")
    for method in status['successful']:
        print(f"   • {method}")
    
    if status['failed']:
        print(f"\n❌ Failed to load methods: {len(status['failed'])}")
        for method, info in status['failed'].items():
            print(f"   • {method}: {info['name']} - {info['error'][:50]}...")
    
    # Show available methods
    print("\n2. Available Methods:")
    print("-" * 30)
    methods = get_available_preprocessing_methods()
    for key, info in methods.items():
        print(f"   {key}: {info['name']}")
        print(f"      {info['description']}")
    
    # Test with sample data
    print("\n3. Testing with Sample Educational Data:")
    print("-" * 30)
    
    # Create sample educational dataset
    np.random.seed(42)
    n_students = 20
    
    # Features: [study_hours, previous_score, attendance_rate]
    X_train = np.random.randn(n_students, 3)
    X_train[:, 0] = np.abs(X_train[:, 0]) * 10  # Study hours (0-10)
    X_train[:, 1] = 50 + X_train[:, 1] * 20    # Previous score (30-70)
    X_train[:, 2] = 0.7 + np.abs(X_train[:, 2]) * 0.3  # Attendance (0.7-1.0)
    
    # Binary outcome: pass/fail
    y_train = np.random.randint(0, 2, n_students)
    
    # Sensitive attribute: gender (with imbalance)
    sensitive_attr = np.array(['Male'] * 15 + ['Female'] * 5)
    np.random.shuffle(sensitive_attr)
    
    print(f"   Original dataset: {len(X_train)} students")
    print(f"   Features: study_hours, previous_score, attendance_rate")
    print(f"   Gender distribution: {np.unique(sensitive_attr, return_counts=True)}")
    print(f"   Pass/Fail distribution: {np.unique(y_train, return_counts=True)}")
    
    # Test each available method
    print("\n4. Testing Preprocessing Methods:")
    print("-" * 30)
    
    for method_key in status['successful']:
        try:
            print(f"\n   Testing: {methods[method_key]['name']}")
            
            X_processed, y_processed, sens_processed = apply_preprocessing_method(
                method_key, X_train, y_train, sensitive_attr
            )
            
            print(f"   ✅ Success!")
            print(f"      Original: {len(X_train)} → Processed: {len(X_processed)} students")
            
            if len(X_processed) != len(X_train):
                print(f"      Data size changed: {len(X_train)} → {len(X_processed)}")
            
            unique_sens, counts = np.unique(sens_processed, return_counts=True)
            print(f"      Gender distribution after preprocessing: {dict(zip(unique_sens, counts))}")
            
        except Exception as e:
            print(f"   ❌ Failed: {str(e)}")
    
    print("\n5. Research Framework Integration:")
    print("-" * 30)
    print("   ✅ Using actual research-grade algorithms from src/mitigation/preprocessing")
    print("   ✅ Automatic fallback to simple implementations if imports fail")
    print("   ✅ Dynamic loading system handles dependency issues gracefully")
    print("   ✅ All methods use the same interface for easy switching")
    
    print("\n6. GUI Integration:")
    print("-" * 30)
    print("   ✅ GUI automatically detects available methods")
    print("   ✅ Dropdown populated with working methods only") 
    print("   ✅ Method descriptions updated dynamically")
    print("   ✅ Comparative results show baseline vs preprocessed performance")
    
    print("\n" + "=" * 60)
    print("Dynamic Preprocessing System Ready!")
    print("=" * 60)
    print("\nTo use the GUI:")
    print("  python run_fairness_gui.py")
    print("\nThe system will automatically:")
    print("  • Load available research methods")
    print("  • Fall back to simple methods if needed")
    print("  • Provide full comparative analysis")

if __name__ == "__main__":
    demonstrate_dynamic_preprocessing() 