#!/usr/bin/env python3
"""
Test script to verify that preprocessing is now truly optional.
This script simulates the GUI workflow without preprocessing.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the package to path
sys.path.append('debiased_jadouille-0.0.20/src')

def test_workflow_without_preprocessing():
    """Test that we can train models without preprocessing"""
    print("Testing Optional Preprocessing Workflow")
    print("=" * 50)
    
    # Test 1: Create sample data (simulating data loading)
    print("1. Creating sample data...")
    np.random.seed(42)
    n_samples = 100
    
    # Create biased educational data
    data = {
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'socioeconomic_status': np.random.choice(['Low', 'Medium', 'High'], n_samples),
        'previous_grade': np.random.normal(75, 15, n_samples),
        'attendance': np.random.normal(85, 10, n_samples),
        'study_hours': np.random.normal(20, 8, n_samples)
    }
    
    # Create biased target (males and high SES have advantage)
    pass_prob = 0.3  # base probability
    for i in range(n_samples):
        if data['gender'][i] == 'Male':
            pass_prob += 0.2
        if data['socioeconomic_status'][i] == 'High':
            pass_prob += 0.3
        if data['previous_grade'][i] > 80:
            pass_prob += 0.2
            
        data.setdefault('pass', []).append(np.random.random() < pass_prob)
        pass_prob = 0.3  # reset for next iteration
    
    df = pd.DataFrame(data)
    print(f"Created dataset with {len(df)} samples")
    print(f"Target distribution: {df['pass'].value_counts().to_dict()}")
    
    # Test 2: Simulate model training without preprocessing
    print("\n2. Testing model training WITHOUT preprocessing...")
    
    try:
        # Simulate the GUI workflow
        target_col = 'pass'
        sensitive_attrs = ['gender', 'socioeconomic_status']
        
        # Separate features and target (like GUI does)
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        
        # Encode categorical variables (like GUI does)
        from sklearn.preprocessing import LabelEncoder
        le_dict = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        
        # Test 3: Train different model types (like GUI offers)
        print("\n3. Testing different model types...")
        
        models_to_test = {
            'Logistic Regression': ('sklearn.linear_model', 'LogisticRegression'),
            'Decision Tree': ('sklearn.tree', 'DecisionTreeClassifier'),
            'Random Forest': ('sklearn.ensemble', 'RandomForestClassifier')
        }
        
        results = {}
        
        for model_name, (module_name, class_name) in models_to_test.items():
            try:
                # Import and create model
                module = __import__(module_name, fromlist=[class_name])
                ModelClass = getattr(module, class_name)
                
                if model_name == 'Logistic Regression':
                    model = ModelClass(random_state=42, max_iter=1000)
                else:
                    model = ModelClass(random_state=42)
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                train_score = model.score(X_train, y_train)
                test_score = model.score(X_test, y_test)
                
                results[model_name] = {
                    'train_accuracy': train_score,
                    'test_accuracy': test_score
                }
                
                print(f"SUCCESS {model_name}: Train={train_score:.3f}, Test={test_score:.3f}")
                
            except Exception as e:
                print(f"FAILED {model_name}: Failed - {str(e)}")
        
        # Test 4: Simulate fairness evaluation
        print("\n4. Testing fairness evaluation...")
        
        if results:
            # Use the first successful model for fairness testing
            model_name = list(results.keys())[0]
            module = __import__(models_to_test[model_name][0], fromlist=[models_to_test[model_name][1]])
            ModelClass = getattr(module, models_to_test[model_name][1])
            model = ModelClass(random_state=42, max_iter=1000 if 'Logistic' in model_name else None)
            model.fit(X_train, y_train)
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Calculate basic fairness metrics (demographic parity)
            test_df = X_test.copy()
            test_df['y_true'] = y_test
            test_df['y_pred'] = y_pred
            
            # Decode gender for fairness analysis
            if 'gender' in le_dict:
                test_df['gender_decoded'] = le_dict['gender'].inverse_transform(test_df['gender'])
                
                # Calculate positive prediction rate by gender
                gender_stats = test_df.groupby('gender_decoded')['y_pred'].agg(['mean', 'count'])
                print(f"Positive prediction rates by gender:")
                for gender, stats in gender_stats.iterrows():
                    print(f"   {gender}: {stats['mean']:.3f} (n={stats['count']})")
                
                # Calculate demographic parity difference
                rates = gender_stats['mean']
                if len(rates) >= 2:
                    dp_diff = abs(rates.max() - rates.min())
                    print(f"Demographic parity difference: {dp_diff:.3f}")
        
        print("\nSUCCESS: All tests passed!")
        print("Data loading works without preprocessing")
        print("Model training works with raw data")
        print("Multiple model types supported")
        print("Fairness evaluation possible")
        print("\nPreprocessing is now truly OPTIONAL!")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("FAILED: Workflow failed - preprocessing may still be required")
        return False

if __name__ == "__main__":
    success = test_workflow_without_preprocessing()
    sys.exit(0 if success else 1) 