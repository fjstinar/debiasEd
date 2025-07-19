# Dataset Preparation Guide for DebiasEd

This guide shows you **step-by-step** how to prepare your own educational datasets for bias analysis in the DebiasEd system.

## Quick Overview

The DebiasEd system expects datasets in a specific format:
- **File**: `data_dictionary.pkl` (Python pickle file)
- **Location**: `data/your_dataset_name/`
- **Format**: Dictionary with student records and demographic information

## Required Data Format

Your `data_dictionary.pkl` must contain:

```python
{
    'data': {
        0: {
            'learner_id': 0,
            'features': [1.2, 3.4, 5.6, ...],      # Numerical features for ML
            'binary_label': 1,                      # Target: 0 or 1
            'gender': 0,                            # Demographic attributes
            'age': 2,                               # (encoded as integers)
            'ethnicity': 1
        },
        1: { ... },
        # ... more records
    },
    'available_demographics': ['gender', 'age', 'ethnicity']
}
```

## Method 1: Use the Automated Template

### Step 1: Run the preparation script
```python
python dataset_preparation_guide.py
```

### Step 2: Use the template function
```python
from dataset_preparation_guide import prepare_dataset_template

# Prepare your dataset
data_dict = prepare_dataset_template(
    csv_path='your_data.csv',
    dataset_name='my_school_data',
    feature_columns=['math_score', 'reading_score', 'attendance'],
    target_column='pass_fail',
    demographic_columns=['gender', 'ethnicity', 'socioeconomic_status'],
    categorical_mappings={
        'gender': {'Male': 0, 'Female': 1},
        'ethnicity': {'A': 0, 'B': 1, 'C': 2},
        'pass_fail': {'Fail': 0, 'Pass': 1}
    }
)
```

## Method 2: Manual Preparation

### Step 1: Prepare your CSV data
Make sure your CSV has:
- **Features**: Numerical columns for machine learning (test scores, attendance, etc.)
- **Target**: Binary outcome (pass/fail, high/low performance, etc.)
- **Demographics**: Sensitive attributes (gender, race, age, etc.)

### Step 2: Create the preparation script

```python
import pandas as pd
import pickle
import os

# Load your data
df = pd.read_csv('your_data.csv')

# Encode categorical variables
mappings = {
    'gender': {'M': 0, 'F': 1},
    'grade_level': {'Elementary': 0, 'Middle': 1, 'High': 2},
    # ... add more mappings
}

for column, mapping in mappings.items():
    df[column] = df[column].map(mapping)

# Create binary target if needed
df['binary_label'] = (df['final_grade'] >= 70).astype(int)  # Pass/fail at 70%

# Define your columns
feature_columns = ['math_score', 'reading_score', 'attendance_rate']
demographic_columns = ['gender', 'ethnicity', 'grade_level']

# Create data dictionary
data_dict = {
    'data': {},
    'available_demographics': demographic_columns
}

# Populate records
for idx, row in df.iterrows():
    data_dict['data'][idx] = {
        'learner_id': idx,
        'features': row[feature_columns].values.tolist(),
        'binary_label': int(row['binary_label']),
        # Add demographics
        'gender': int(row['gender']),
        'ethnicity': int(row['ethnicity']),
        'grade_level': int(row['grade_level'])
    }

# Save the data dictionary
os.makedirs('data/my_dataset', exist_ok=True)
with open('data/my_dataset/data_dictionary.pkl', 'wb') as fp:
    pickle.dump(data_dict, fp)

print("Dataset prepared successfully!")
```

## Directory Structure

After preparation, your structure should look like:

```
data/
├── my_dataset/
│   └── data_dictionary.pkl     # Your prepared data
├── eedi/                       # Example existing datasets
│   └── data_dictionary.pkl
├── oulad/
│   └── data_dictionary.pkl
└── student-performance-math/
    └── data_dictionary.pkl
```

## Step 3: Validate Your Dataset

Use the validation function to check your format:

```python
from dataset_preparation_guide import validate_data_dictionary
import pickle

# Load and validate
with open('data/my_dataset/data_dictionary.pkl', 'rb') as fp:
    data_dict = pickle.load(fp)

validate_data_dictionary(data_dict)
```

## Step 4: Integrate with DebiasEd

### Option A: Use with GUI (Automatic)
Your dataset will automatically appear in the GUI if properly formatted and placed in the correct directory.

### Option B: Add to Feature Pipeline (Advanced)
For full framework integration, add your dataset to `src/pipelines/feature_pipeline.py`:

```python
# In _select_dataset method:
if self._settings['pipeline']['dataset'] == 'my_dataset':
    path = '{}/my_dataset/data_dictionary.pkl'.format(self._settings['paths']['data'])
    label = 'binary_label'
    self._settings['pipeline']['nclasses'] = 2
    self._settings['pipeline']['attributes'] = {
        'mitigating': 'gender',           # Sensitive attribute
        'discriminated': '_1',            # Protected group (e.g., female=1)
        'included': []
    }

# In _get_stratification_column method:
elif self._settings['pipeline']['dataset'] == 'my_dataset':
    self._settings['crossvalidation']['stratifier_col'] = 'binary_label'
```

## Example Datasets

### Student Performance Example
```python
# Features: test scores, homework completion, attendance
features = [85.5, 92.0, 0.95]  # Math score, reading score, attendance rate

# Target: pass/fail based on final grade  
binary_label = 1  # 1 = pass, 0 = fail

# Demographics: encoded as integers
gender = 1        # 0 = male, 1 = female  
ethnicity = 2     # 0 = A, 1 = B, 2 = C, etc.
ses = 1          # 0 = low, 1 = medium, 2 = high socioeconomic status
```

### Course Completion Example
```python
# Features: engagement metrics, prior knowledge, time spent
features = [75.2, 45.8, 120.5]  # Quiz average, forum posts, hours logged

# Target: course completion
binary_label = 0  # 0 = dropped out, 1 = completed

# Demographics  
age_group = 1     # 0 = young, 1 = middle, 2 = older
education = 0     # 0 = high school, 1 = bachelor, 2 = graduate
location = 3      # Geographic region codes
```

## Checklist

Before using your dataset, verify:

- [ ] **File location**: `data/dataset_name/data_dictionary.pkl`
- [ ] **Required keys**: `'data'` and `'available_demographics'`
- [ ] **Record format**: Each record has `learner_id`, `features`, `binary_label`
- [ ] **Numerical features**: All features are numbers (not strings)
- [ ] **Binary target**: Labels are 0 or 1 only
- [ ] **Encoded demographics**: All demographic values are integers
- [ ] **No missing data**: All required fields are present
- [ ] **Validation passed**: Used validation function successfully

## Troubleshooting

### "No datasets found" in GUI
- Check file path: `data/your_dataset/data_dictionary.pkl`
- Verify file permissions (readable)
- Run from project root directory

### "KeyError" when loading dataset
- Check required keys: `'data'`, `'available_demographics'`
- Validate record format with validation function

### Features not working in ML
- Ensure all features are numerical
- Check for NaN or infinite values
- Consider feature scaling if needed

### Fairness analysis not working
- Verify demographic attributes are properly encoded
- Check that protected groups are defined correctly
- Ensure adequate representation of different groups

## Educational Dataset Ideas

Consider preparing datasets from:
- **Student assessments** (standardized test scores)
- **Course completion** data (MOOCs, online learning)
- **School performance** metrics (graduation rates, GPA)
- **Learning analytics** (engagement, time-on-task)
- **Admissions data** (acceptance/rejection decisions)
- **Teacher evaluations** (performance ratings) 