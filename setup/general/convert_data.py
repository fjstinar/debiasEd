import os
import pickle
import argparse
import numpy as np
import pandas as pd


def auto_detect_columns(df):
    """
    Automatically detect demographics, features, and target columns
    """
    columns = list(df.columns)
    
    # Check if we have the expected format first
    explicit_features = [f for f in columns if f.startswith('feature ')]
    explicit_demographics = [d for d in columns if d.startswith('demo ')]
    
    if explicit_features and explicit_demographics and 'label' in columns:
        print("Found explicit column format with prefixes")
        return explicit_features, explicit_demographics, 'label'
    
    # Auto-detection for real-world datasets
    print("Auto-detecting column types...")
    
    # Common demographic indicators
    demo_keywords = ['sex', 'gender', 'age', 'school', 'address', 'family', 'parent', 'Pstatus']
    demographics = []
    
    # Common target indicators (usually at the end, numerical, named grade/score/result)
    target_keywords = ['label', 'target', 'grade', 'score', 'result', 'G3', 'final']
    target_column = None
    
    # Find target column
    for keyword in target_keywords:
        if keyword in columns:
            target_column = keyword
            break
    
    # If no explicit target found, use last column as target
    if not target_column:
        target_column = columns[-1]
        print(f"Using last column '{target_column}' as target")
    
    # Find demographics
    for col in columns:
        if col == target_column:
            continue
        for keyword in demo_keywords:
            if keyword.lower() in col.lower():
                demographics.append(col)
                break
    
    # Remaining columns (except target) are features
    features = [col for col in columns if col != target_column and col not in demographics]
    
    print(f"Detected {len(demographics)} demographic columns: {demographics}")
    print(f"Detected {len(features)} feature columns: {features[:5]}..." if len(features) > 5 else f"Detected {len(features)} feature columns: {features}")
    print(f"Target column: {target_column}")
    
    return features, demographics, target_column


def convert_csv(path, target_col=None, demo_cols=None, feature_cols=None):
    """
    Convert CSV to pickle format with flexible column detection
    """
    old = pd.read_csv(path)
    print(f"Loaded CSV with {len(old)} rows and {len(old.columns)} columns")
    
    if target_col or demo_cols or feature_cols:
        # Manual specification
        print("Using manually specified columns...")
        demographics = demo_cols or []
        features = feature_cols or []
        target_column = target_col or 'label'
    else:
        # Auto-detection
        features, demographics, target_column = auto_detect_columns(old)
    
    # Validate target column exists
    if target_column not in old.columns:
        raise ValueError(f"Target column '{target_column}' not found in data!")
    
    # Convert data
    new = {'available_demographics': demographics, 'data': {}}
    
    for i, row in old.iterrows():
        new['data'][i] = {demo: row[demo] for demo in demographics}
        new['data'][i]['learner_id'] = i
        new['data'][i]['target'] = row[target_column]
        
        # Handle features - convert to numpy array
        if features:
            feature_values = []
            for f in features:
                val = row[f]
                # Convert categorical to numeric if needed
                if isinstance(val, str):
                    # Simple encoding for yes/no values
                    if val.lower() in ['yes', 'y', 'true']:
                        val = 1
                    elif val.lower() in ['no', 'n', 'false']:
                        val = 0
                    else:
                        # For other strings, use hash (not ideal but works for basic conversion)
                        val = hash(val) % 1000
                feature_values.append(float(val))
            new['data'][i]['features'] = np.array(feature_values)
        else:
            new['data'][i]['features'] = np.array([])

    print(f"Converted {len(new['data'])} records successfully")
    return new

def save(settings, new):
    if settings['name'] == '':
        name = 'personal_data'
    else:
        name = settings['name']
    folder = 'data/{}/'.format(name)
    os.makedirs(folder, exist_ok=True)
    
    output_file = 'data/{}/data_dictionary.pkl'.format(name)
    with open(output_file, 'wb') as fp:
        pickle.dump(new, fp)
    
    print(f"Saved converted data to: {output_file}")
    print(f"Dataset contains:")
    print(f"  - {len(new['data'])} records")
    print(f"  - {len(new['available_demographics'])} demographic attributes: {new['available_demographics']}")
    if len(new['data']) > 0:
        sample_features = new['data'][0]['features']
        print(f"  - {len(sample_features)} feature dimensions")
    print(f"  - Dataset name: '{name}'")


def main(settings):
    # Prepare column specifications
    target_col = settings.get('target')
    demo_cols = settings.get('demographics')
    feature_cols = settings.get('features')
    
    # Parse comma-separated column lists
    if demo_cols:
        demo_cols = [col.strip() for col in demo_cols.split(',')]
    if feature_cols:
        feature_cols = [col.strip() for col in feature_cols.split(',')]
    
    print("=" * 60)
    print("DebiasEd Data Converter")
    print("=" * 60)
    print(f"Input file: {settings['path']}")
    print()
    
    new_data = convert_csv(settings['path'], target_col, demo_cols, feature_cols)
    save(settings, new_data)
    
    print()
    print("✅ Conversion completed successfully!")

if __name__ == '__main__': 
    settings = {}
    parser = argparse.ArgumentParser(
        description='Convert CSV data to DebiasEd pickle format',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect columns (recommended for student performance data)
  python convert_data.py --path student-mat_converted.csv --name student_math

  # Manual column specification
  python convert_data.py --path data.csv --name mydata --target grade --demographics "sex,age,school"
  
  # Original format (with prefixes)
  python convert_data.py --path data.csv --name dataset
        """
    )
    parser.add_argument('--path', dest='path', required=True,
                      help='Path to input CSV file')
    parser.add_argument('--name', dest='name', default='',
                      help='Name for the converted dataset')
    parser.add_argument('--target', dest='target', 
                      help='Name of target/label column (auto-detected if not specified)')
    parser.add_argument('--demographics', dest='demographics',
                      help='Comma-separated list of demographic columns (auto-detected if not specified)')
    parser.add_argument('--features', dest='features',
                      help='Comma-separated list of feature columns (auto-detected if not specified)')
    
    settings.update(vars(parser.parse_args()))

    if not settings['path']:
        parser.error("--path is required")

    main(settings)