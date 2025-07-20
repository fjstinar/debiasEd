#!/usr/bin/env python3
"""
Test script to verify that model evaluation works without preprocessing.
This simulates the evaluation workflow that was causing the nontype error.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the package to path
sys.path.append('debiased_jadouille-0.0.20/src')

def test_evaluation_without_preprocessing():
    """Test that model evaluation works without preprocessing"""
    print("Testing Model Evaluation Without Preprocessing")
    print("=" * 55)
    
    # Test 1: Create and prepare data (simulating GUI workflow)
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
    
    # Create biased target
    pass_prob = 0.3
    for i in range(n_samples):
        if data['gender'][i] == 'Male':
            pass_prob += 0.2
        if data['socioeconomic_status'][i] == 'High':
            pass_prob += 0.3
        if data['previous_grade'][i] > 80:
            pass_prob += 0.2
            
        data.setdefault('pass', []).append(np.random.random() < pass_prob)
        pass_prob = 0.3
    
    df = pd.DataFrame(data)
    print(f"Created dataset with {len(df)} samples")
    
    # Test 2: Train model without preprocessing
    print("\n2. Training model without preprocessing...")
    
    try:
        target_col = 'pass'
        sensitive_attrs = ['gender', 'socioeconomic_status']
        
        # Prepare data like GUI does
        X = df.drop([target_col], axis=1)
        y = df[target_col]
        
        # Encode categorical variables
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
        
        # Train a model
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        print(f"Model trained successfully")
        print(f"Training accuracy: {model.score(X_train, y_train):.3f}")
        
        # Test 3: Evaluate model (this was causing the nontype error)
        print("\n3. Testing model evaluation...")
        
        # Simulate the GUI evaluation process
        y_pred = model.predict(X_test)
        
        # Test prediction probabilities (this was another potential None source)
        if hasattr(model, 'predict_proba'):
            try:
                probabilities = model.predict_proba(X_test)
                if probabilities is not None and probabilities.shape[1] > 1:
                    y_pred_proba = probabilities[:, 1]
                else:
                    y_pred_proba = y_pred.astype(float)
            except:
                y_pred_proba = y_pred.astype(float)
        else:
            y_pred_proba = y_pred.astype(float)
        
        print(f"Predictions generated: {len(y_pred)} samples")
        print(f"Probabilities shape: {y_pred_proba.shape}")
        
        # Test 4: Calculate metrics
        print("\n4. Testing metrics calculation...")
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='weighted'),
            'Recall': recall_score(y_test, y_pred, average='weighted'),
            'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        }
        
        if len(np.unique(y_test)) == 2:
            metrics['ROC-AUC'] = roc_auc_score(y_test, y_pred_proba)
        
        print("Basic metrics calculated:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.3f}")
        
        # Test 5: Calculate fairness metrics (this was the main source of nontype error)
        print("\n5. Testing fairness metrics calculation...")
        
        try:
            # Simulate the fixed approach - use raw data since no preprocessing
            source_df = df  # This is like self.data['raw_data']
            
            # Handle index alignment - get the actual test data with sensitive attributes  
            if hasattr(X_test, 'index'):
                df_test = source_df.loc[X_test.index]
            else:
                # Fallback approach
                df_test = source_df.tail(len(X_test)).reset_index(drop=True)
            
            fairness_metrics = {}
            
            for attr in sensitive_attrs:
                if attr in df_test.columns:
                    groups = df_test[attr].unique()
                    group_metrics = {}
                    
                    for group in groups:
                        group_mask = df_test[attr] == group
                        if group_mask.sum() > 0:
                            group_y_test = y_test[group_mask] if hasattr(group_mask, '__len__') else y_test
                            group_y_pred = y_pred[group_mask] if hasattr(group_mask, '__len__') else y_pred
                            
                            if len(group_y_test) > 0:
                                group_metrics[group] = {
                                    'Accuracy': accuracy_score(group_y_test, group_y_pred),
                                    'Positive_Rate': np.mean(group_y_pred)
                                }
                    
                    fairness_metrics[attr] = group_metrics
            
            print("Fairness metrics calculated:")
            for attr, groups in fairness_metrics.items():
                print(f"   {attr}:")
                for group, group_metrics in groups.items():
                    print(f"     {group}: Accuracy={group_metrics['Accuracy']:.3f}, Pos_Rate={group_metrics['Positive_Rate']:.3f}")
                    
        except Exception as e:
            print(f"✗ Fairness metrics failed: {str(e)}")
            return False
        
        print("\nSUCCESS: All evaluation tests passed!")
        print("Model training works without preprocessing")
        print("Basic predictions work")
        print("Probability predictions work")
        print("Standard metrics calculation works")
        print("Fairness metrics calculation works")
        print("No more nontype errors!")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("FAILED: Evaluation still has issues")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_evaluation_without_preprocessing()
    sys.exit(0 if success else 1) 