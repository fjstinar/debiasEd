#!/usr/bin/env python3
"""
Comprehensive Preprocessing Test

This script tests all 26+ preprocessing methods now available in the dynamic system
and shows which ones are working vs those that have dependency issues.
"""

import sys
import os
import numpy as np

# Add the gui directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gui'))

from gui.preprocessing_dynamic import get_available_preprocessing_methods, apply_preprocessing_method, get_import_status

def test_all_preprocessing_methods():
    """Test all available preprocessing methods with educational data"""
    
    print("=" * 80)
    print("COMPREHENSIVE PREPROCESSING METHODS TEST")
    print("=" * 80)
    
    # Get import status
    status = get_import_status()
    methods = get_available_preprocessing_methods()
    
    print(f"\nSUMMARY:")
    print(f"   Total methods discovered: {len(methods) - 1}")  # -1 for 'none'
    print(f"   Successfully loaded: {len(status['successful'])}")
    print(f"   Failed to load: {len(status['failed'])}")
    
    # Create realistic educational dataset
    print(f"\nSAMPLE EDUCATIONAL DATASET:")
    np.random.seed(42)
    n_students = 100
    
    # Features: [study_hours, previous_score, attendance_rate, assignment_completion]
    X_train = np.random.randn(n_students, 4)
    X_train[:, 0] = np.abs(X_train[:, 0]) * 5  # Study hours (0-10)
    X_train[:, 1] = 50 + X_train[:, 1] * 15    # Previous score (20-80)
    X_train[:, 2] = 0.7 + np.abs(X_train[:, 2]) * 0.25  # Attendance (0.7-1.0)
    X_train[:, 3] = 0.6 + np.abs(X_train[:, 3]) * 0.35  # Assignment completion (0.6-1.0)
    
    # Binary outcome: pass/fail with some bias
    pass_probability = 0.3 + 0.7 * (X_train[:, 0] > 3) * (X_train[:, 2] > 0.8)
    y_train = np.random.binomial(1, pass_probability, n_students)
    
    # Sensitive attribute: gender (with imbalance to demonstrate bias)
    sensitive_attr = np.array(['Male'] * 70 + ['Female'] * 30)
    np.random.shuffle(sensitive_attr)
    
    print(f"   Students: {len(X_train)}")
    print(f"   Features: study_hours, previous_score, attendance_rate, assignment_completion")
    gender_counts = np.unique(sensitive_attr, return_counts=True)
    print(f"   Gender distribution: {dict(zip(gender_counts[0], gender_counts[1]))}")
    pass_counts = np.unique(y_train, return_counts=True)
    print(f"   Pass/Fail distribution: {dict(zip(['Fail', 'Pass'], pass_counts[1]))}")
    
    # Test working methods
    print(f"\nSUCCESSFULLY WORKING METHODS:")
    print("-" * 50)
    working_count = 0
    
    for method_key in status['successful']:
        if method_key == 'none':
            continue
            
        try:
            X_processed, y_processed, sens_processed = apply_preprocessing_method(
                method_key, X_train, y_train, sensitive_attr
            )
            
            working_count += 1
            
            # Calculate size change
            size_change = len(X_processed) - len(X_train)
            size_indicator = f"→ {len(X_processed)}" if size_change != 0 else "unchanged"
            
            # Calculate gender distribution change
            gender_dist_after = np.unique(sens_processed, return_counts=True)
            gender_dict_after = dict(zip(gender_dist_after[0], gender_dist_after[1]))
            
            print(f"   {working_count:2d}. {methods[method_key]['name']}")
            print(f"       Size: {len(X_train)} {size_indicator}")
            print(f"       Gender after: {gender_dict_after}")
            
        except Exception as e:
            # Method loaded but failed to execute
            print(f"   ❌ {methods[method_key]['name']}: {str(e)[:60]}...")
    
    # Show failed imports
    print(f"\nMETHODS WITH DEPENDENCY ISSUES:")
    print("-" * 50)
    
    for i, (method_key, info) in enumerate(status['failed'].items(), 1):
        error_summary = info['error'].split(':')[0] if ':' in info['error'] else info['error']
        print(f"   {i:2d}. {info['name']}")
        print(f"       Issue: {error_summary}")
    
    # Show method categories
    print(f"\nAVAILABLE METHOD CATEGORIES:")
    print("-" * 50)
    
    categories = {
        'Basic Sampling': ['rebalance', 'calders', 'smote', 'singh'],
        'Kamiran Family': ['kamiran', 'kamiran2'],
        'Zelaya Family': ['zelaya_over', 'zelaya_under', 'zelaya_smote', 'zelaya_psp'],
        'Iosifidis Family': ['iosifidis_smote_attr', 'iosifidis_smote_target', 'iosifidis_resample_attr', 'iosifidis_resample_target'],
        'Representation Learning': ['zemel', 'luong'],
        'Statistical Methods': ['alabdulmohsin', 'salazar', 'yan', 'chakraborty', 'cohausz', 'dablain', 'cock', 'li', 'lahoti', 'jiang']
    }
    
    for category, method_list in categories.items():
        available_in_category = []
        for method in method_list:
            if method in status['successful']:
                available_in_category.append(methods[method]['name'])
        
        print(f"   {category}: {len(available_in_category)} available")
        if available_in_category:
            for method_name in available_in_category[:3]:  # Show first 3
                print(f"     • {method_name}")
            if len(available_in_category) > 3:
                print(f"     • ... and {len(available_in_category) - 3} more")
    
    print(f"\nUSAGE IN GUI:")
    print("-" * 50)
    print("   The GUI will now show a dropdown with all working methods:")
    print(f"   • {working_count} preprocessing methods available")
    print("   • Automatic descriptions for each method")
    print("   • Comparative results showing improvement/degradation")
    print("   • Graceful handling of methods with missing dependencies")
    
    print(f"\n" + "=" * 80)
    print("COMPREHENSIVE PREPROCESSING SYSTEM READY!")
    print("=" * 80)
    print(f"Run 'python run_fairness_gui.py' to use the enhanced GUI")
    print(f"with {working_count} research-grade preprocessing methods!")

if __name__ == "__main__":
    test_all_preprocessing_methods() 