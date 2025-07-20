#!/usr/bin/env python3
"""
Test script to verify that preprocessing methods are actually being applied
"""

import pandas as pd
import numpy as np
import sys

# Add the package to the path
sys.path.append('debiased_jadouille-0.0.20/src')

def test_preprocessing_import():
    """Test if we can import the preprocessing classes"""
    print("Testing preprocessing imports...")
    
    try:
        from debiased_jadouille.mitigation.preprocessing.calders import CaldersPreProcessor
        print("CaldersPreProcessor imported successfully")
        
        from debiased_jadouille.mitigation.preprocessing.smote import SmotePreProcessor
        print("SmotePreProcessor imported successfully")
        
        from debiased_jadouille.mitigation.preprocessing.zemel import ZemelPreProcessor
        print("ZemelPreProcessor imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_calders_preprocessing():
    """Test Calders preprocessing method"""
    print("\nTesting Calders preprocessing...")
    
    try:
        from debiased_jadouille.mitigation.preprocessing.calders import CaldersPreProcessor
        
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        # Create biased data
        gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.7, 0.3])
        features = np.random.randn(n_samples, 3)
        
        # Create biased target
        target = np.zeros(n_samples)
        for i in range(n_samples):
            if gender[i] == 'Male':
                target[i] = np.random.choice([0, 1], p=[0.3, 0.7])  # Males more likely to pass
            else:
                target[i] = np.random.choice([0, 1], p=[0.6, 0.4])  # Females less likely to pass
        
        # Prepare data for preprocessing
        X = features.tolist()
        y = target.tolist()
        demo_data = [{'gender': g} for g in gender]
        
        print(f"Original data: {len(X)} samples")
        print(f"Original gender distribution: {pd.Series(gender).value_counts().to_dict()}")
        print(f"Original pass rates by gender:")
        df_orig = pd.DataFrame({'gender': gender, 'pass': target})
        for g in df_orig['gender'].unique():
            rate = df_orig[df_orig['gender'] == g]['pass'].mean()
            print(f"  {g}: {rate:.3f}")
        
        # Apply Calders preprocessing
        preprocessor = CaldersPreProcessor('gender', None, 1.0)
        X_processed, y_processed, demo_processed = preprocessor.fit_transform(X, y, demo_data)
        
        print(f"\nProcessed data: {len(X_processed)} samples")
        
        # Check if preprocessing had an effect
        if len(X_processed) != len(X) or np.array_equal(y_processed, y):
            print("Warning: Preprocessing may not have had a significant effect")
        else:
            print("Preprocessing appears to have modified the data")
            
        return True
        
    except Exception as e:
        print(f"✗ Calders preprocessing failed: {e}")
        return False

def create_test_data():
    """Create test data with bias for GUI testing"""
    print("\nCreating test data with bias...")
    
    np.random.seed(42)
    n_samples = 200
    
    # Generate biased educational data
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    ethnicity = np.random.choice(['White', 'Black', 'Hispanic'], n_samples, p=[0.6, 0.2, 0.2])
    age = np.random.normal(20, 2, n_samples)
    previous_grade = np.random.normal(75, 10, n_samples)
    study_hours = np.random.exponential(5, n_samples)
    
    # Create biased target variable
    pass_outcome = np.zeros(n_samples)
    for i in range(n_samples):
        base_prob = 0.5
        
        # Gender bias
        if gender[i] == 'Male':
            base_prob += 0.15
        else:
            base_prob -= 0.15
            
        # Ethnicity bias
        if ethnicity[i] == 'White':
            base_prob += 0.1
        elif ethnicity[i] == 'Black':
            base_prob -= 0.1
            
        # Feature-based probability
        feature_effect = (previous_grade[i] - 75) * 0.01 + study_hours[i] * 0.02
        final_prob = base_prob + feature_effect * 0.1
        final_prob = np.clip(final_prob, 0.1, 0.9)
        
        pass_outcome[i] = np.random.binomial(1, final_prob)
    
    # Create DataFrame
    test_data = pd.DataFrame({
        'gender': gender,
        'ethnicity': ethnicity,
        'age': np.round(age, 1),
        'previous_grade': np.round(previous_grade, 1),
        'study_hours': np.round(study_hours, 1),
        'pass': pass_outcome.astype(int)
    })
    
    # Save to CSV
    test_data.to_csv('test_biased_data.csv', index=False)
    
    # Print bias analysis
    print(f"Test data created: {len(test_data)} samples")
    print("Bias analysis:")
    overall_pass_rate = test_data['pass'].mean()
    print(f"Overall pass rate: {overall_pass_rate:.3f}")
    
    for gender_val in test_data['gender'].unique():
        rate = test_data[test_data['gender'] == gender_val]['pass'].mean()
        bias = rate - overall_pass_rate
        print(f"  {gender_val}: {rate:.3f} (bias: {bias:+.3f})")
    
    print("Test data saved to 'test_biased_data.csv'")
    print("You can load this file in the GUI to test preprocessing!")
    
def main():
    """Main test function"""
    print("=== Preprocessing Test Suite ===")
    
    # Test 1: Import test
    import_success = test_preprocessing_import()
    
    # Test 2: Actual preprocessing test
    if import_success:
        preprocessing_success = test_calders_preprocessing()
    else:
        print("Skipping preprocessing test due to import failure")
    
    # Test 3: Create test data
    create_test_data()
    
    print("\n=== Test Summary ===")
    if import_success:
        print("Package imports working")
        print("You can now test real preprocessing in the GUI")
    else:
        print("Warning: Package imports failed - GUI will use simulation mode")
    
    print("Test data created for GUI testing")
    print("\nNext steps:")
    print("1. Launch the GUI: python run_gui.py")
    print("2. Load 'test_biased_data.csv'")
    print("3. Select 'gender' as sensitive attribute")
    print("4. Select 'pass' as target column")
    print("5. Try different preprocessing methods")
    print("6. Check the Before/After/Comparison tabs!")

if __name__ == "__main__":
    main() 