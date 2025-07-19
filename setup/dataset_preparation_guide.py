#!/usr/bin/env python3
"""
DebiasEd Dataset Preparation Guide
==================================

This guide shows you how to prepare your own datasets for the DebiasEd fairness system.
The system expects data in a specific pickle format called 'data_dictionary.pkl'.

QUICK START:
1. Follow the template below
2. Adapt it to your data
3. Save as data_dictionary.pkl in data/your_dataset_name/
4. Update the feature pipeline if needed

REQUIRED FORMAT:
Your data_dictionary.pkl should contain a dictionary with:
- 'data': Dictionary of records indexed by learner ID
- 'available_demographics': List of demographic attribute names

Each record in 'data' should have:
- 'learner_id': Unique identifier
- 'features': List/array of numerical features for ML
- 'binary_label': Target variable (0 or 1)
- Demographics: One or more demographic attributes (gender, age, etc.)
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def prepare_dataset_template(csv_path, dataset_name, 
                           feature_columns, 
                           target_column, 
                           demographic_columns,
                           categorical_mappings=None):
    """
    Template function to prepare any dataset for DebiasEd
    
    Args:
        csv_path: Path to your CSV file
        dataset_name: Name for your dataset (creates folder)
        feature_columns: List of column names to use as ML features  
        target_column: Name of the target variable column
        demographic_columns: List of demographic attribute columns
        categorical_mappings: Optional dict of column->value mappings
    
    Returns:
        Dictionary in the required format
    """
    
    print(f"Preparing dataset: {dataset_name}")
    print(f"Reading data from: {csv_path}")
    
    # 1. Load your data
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records with {len(df.columns)} columns")
    
    # 2. Handle missing values (customize as needed)
    initial_size = len(df)
    df = df.dropna(subset=feature_columns + [target_column] + demographic_columns)
    print(f"Removed {initial_size - len(df)} records with missing values")
    
    # 3. Encode categorical variables
    if categorical_mappings:
        for column, mapping in categorical_mappings.items():
            if column in df.columns:
                df[column] = df[column].map(mapping)
                print(f"Encoded {column}: {mapping}")
    
    # Auto-encode remaining categorical columns
    for col in feature_columns + demographic_columns:
        if df[col].dtype == 'object':  # String/categorical
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            mapping = dict(zip(le.classes_, le.transform(le.classes_)))
            print(f"Auto-encoded {col}: {mapping}")
    
    # 4. Create binary target if needed
    if df[target_column].nunique() > 2:
        # For regression-like targets, create binary split at median
        median_val = df[target_column].median()
        df['binary_label'] = (df[target_column] > median_val).astype(int)
        print(f"Created binary target: 0 (<= {median_val}), 1 (> {median_val})")
    else:
        # Already binary, just rename
        df['binary_label'] = df[target_column].astype(int)
        print(f"Using binary target: {df['binary_label'].value_counts().to_dict()}")
    
    # 5. Prepare features (standardization optional)
    feature_data = df[feature_columns].values
    print(f"Features shape: {feature_data.shape}")
    
    # 6. Create the data dictionary
    data_dict = {
        'data': {},
        'available_demographics': demographic_columns
    }
    
    # 7. Populate data records
    for idx, (_, row) in enumerate(df.iterrows()):
        record = {
            'learner_id': idx,
            'features': row[feature_columns].values.tolist(),
            'binary_label': int(row['binary_label'])
        }
        
        # Add demographic attributes
        for demo_col in demographic_columns:
            record[demo_col] = int(row[demo_col])
        
        data_dict['data'][idx] = record
    
    print(f"Created {len(data_dict['data'])} data records")
    print(f"Demographics: {demographic_columns}")
    
    # 8. Save the data dictionary
    output_dir = f"./ƒdata/{dataset_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/data_dictionary.pkl"
    
    with open(output_path, 'wb') as fp:
        pickle.dump(data_dict, fp)
    
    print(f"Saved to: {output_path}")
    return data_dict

def example_student_grades():
    """
    Example: Preparing a student grades dataset
    Shows how to adapt the template to your specific data
    """
    
    # Create example data (replace with your CSV path)
    example_data = {
        'student_id': range(1000),
        'math_score': np.random.normal(75, 15, 1000),
        'reading_score': np.random.normal(80, 12, 1000), 
        'writing_score': np.random.normal(78, 14, 1000),
        'gender': np.random.choice(['M', 'F'], 1000),
        'ethnicity': np.random.choice(['A', 'B', 'C', 'D', 'E'], 1000),
        'parental_education': np.random.choice(['high_school', 'some_college', 'bachelor', 'master'], 1000),
        'final_grade': np.random.choice(['A', 'B', 'C', 'D', 'F'], 1000)
    }
    
    df = pd.DataFrame(example_data)
    df.to_csv('example_student_data.csv', index=False)
    
    # Define your dataset configuration
    categorical_mappings = {
        'gender': {'M': 0, 'F': 1},
        'ethnicity': {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4},
        'parental_education': {'high_school': 0, 'some_college': 1, 'bachelor': 2, 'master': 3},
        'final_grade': {'F': 0, 'D': 0, 'C': 0, 'B': 1, 'A': 1}  # Binary: pass/fail
    }
    
    # Prepare the dataset
    data_dict = prepare_dataset_template(
        csv_path='example_student_data.csv',
        dataset_name='student_grades_example',
        feature_columns=['math_score', 'reading_score', 'writing_score'],
        target_column='final_grade', 
        demographic_columns=['gender', 'ethnicity', 'parental_education'],
        categorical_mappings=categorical_mappings
    )
    
    return data_dict

def validate_data_dictionary(data_dict):
    """
    Validate that your data dictionary meets DebiasEd requirements
    """
    print("Validating data dictionary format...")
    
    errors = []
    warnings = []
    
    # Required keys
    if 'data' not in data_dict:
        errors.append("Missing 'data' key")
    if 'available_demographics' not in data_dict:
        errors.append("Missing 'available_demographics' key")
    
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    # Check data records
    sample_records = list(data_dict['data'].values())[:5]
    required_fields = ['learner_id', 'features', 'binary_label']
    
    for i, record in enumerate(sample_records):
        for field in required_fields:
            if field not in record:
                errors.append(f"Record {i} missing '{field}'")
        
        # Check features format
        if 'features' in record and not isinstance(record['features'], (list, np.ndarray)):
            warnings.append(f"Record {i} features should be list/array")
        
        # Check binary label
        if 'binary_label' in record and record['binary_label'] not in [0, 1]:
            warnings.append(f"Record {i} binary_label should be 0 or 1")
    
    # Check demographics
    for demo in data_dict['available_demographics']:
        if demo not in sample_records[0]:
            warnings.append(f"Demographic '{demo}' not found in records")
    
    if errors:
        print("Validation failed:")
        for error in errors:
            print(f"   - {error}")
        return False
    
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"   - {warning}")
    
    print("Validation passed!")
    print(f"Dataset summary:")
    print(f"   - Records: {len(data_dict['data'])}")
    print(f"   - Demographics: {data_dict['available_demographics']}")
    print(f"   - Feature count: {len(sample_records[0]['features']) if sample_records else 'Unknown'}")
    
    return True

def add_to_feature_pipeline(dataset_name, sensitive_attribute, discriminated_group):
    """
    Generate code to add your dataset to the feature pipeline
    """
    
    code_template = f'''
# Add this to src/pipelines/feature_pipeline.py in _select_dataset method:

if self._settings['pipeline']['dataset'] == '{dataset_name}':
    path = '{{}}/' + '{dataset_name}/data_dictionary.pkl'.format(self._settings['paths']['data'])
    label = 'binary_label'
    self._settings['pipeline']['nclasses'] = 2
    self._settings['pipeline']['attributes'] = {{
        'mitigating': '{sensitive_attribute}',
        'discriminated': '{discriminated_group}',  # Define your protected group
        'included': []
    }}

# Add this to _get_stratification_column method:
elif self._settings['pipeline']['dataset'] == '{dataset_name}':
    self._settings['crossvalidation']['stratifier_col'] = 'binary_label'
    '''
    
    print("📝 Feature Pipeline Integration:")
    print(code_template)
    
    return code_template

if __name__ == "__main__":
    print("DebiasEd Dataset Preparation Guide")
    print("=" * 50)
    
    print("\n1. This guide shows you how to prepare datasets")
    print("2. Run example_student_grades() to see a complete example")  
    print("3. Use validate_data_dictionary() to check your format")
    print("4. Use add_to_feature_pipeline() for integration")
    
    print("\nQuick example:")
    print("   data_dict = example_student_grades()")
    print("   validate_data_dictionary(data_dict)")
    
    # Uncomment to run example
    # example_student_grades() 