# Quick Start: Prepare Your Dataset (5 Minutes)

Got your own educational data? Here's how to get it into DebiasEd in **5 simple steps**:

## Super Quick Method

### 1. Install dependencies (if needed)
```bash
pip install pandas numpy scikit-learn
```

### 2. Run the auto-preparation script
```python
from dataset_preparation_guide import prepare_dataset_template

# Just replace these with your actual values:
prepare_dataset_template(
    csv_path='YOUR_DATA.csv',                    # Path to your CSV file
    dataset_name='my_school_data',               # Choose a name
    feature_columns=['score1', 'score2'],       # Columns for ML prediction  
    target_column='final_grade',                 # What you want to predict
    demographic_columns=['gender', 'race'],     # Sensitive attributes
    categorical_mappings={                       # How to encode categories
        'gender': {'Male': 0, 'Female': 1},
        'final_grade': {'Fail': 0, 'Pass': 1}
    }
)
```

### 3. Test it works
```bash
python run_fairness_gui.py --check
```
You should see: `Found X datasets: my_school_data`

### 4. Launch the GUI
```bash
python run_fairness_gui.py
```

### 5. Click "Load my_school_data" and explore!

---

## Your Data Should Look Like This

**CSV Format (before processing):**
```csv
student_id,math_score,reading_score,gender,ethnicity,final_grade
1,85,92,Female,Hispanic,Pass
2,78,85,Male,White,Pass  
3,65,70,Female,Black,Fail
...
```

**Requirements:**
- **Features**: Numerical columns (test scores, attendance, etc.)
- **Target**: What you want to predict (pass/fail, high/low performance)
- **Demographics**: Sensitive attributes (gender, race, age, etc.)
- **Clean data**: No excessive missing values

---

## Quick Examples

### Example 1: Student Test Scores
```python
prepare_dataset_template(
    csv_path='student_scores.csv',
    dataset_name='test_scores',
    feature_columns=['math', 'reading', 'science'],
    target_column='passed_grade',
    demographic_columns=['gender', 'ethnicity', 'free_lunch'],
    categorical_mappings={
        'gender': {'M': 0, 'F': 1},
        'passed_grade': {'No': 0, 'Yes': 1}
    }
)
```

### Example 2: Course Completion
```python  
prepare_dataset_template(
    csv_path='course_data.csv',
    dataset_name='course_completion',
    feature_columns=['quiz_avg', 'forum_posts', 'time_spent'],
    target_column='completed',
    demographic_columns=['age_group', 'education_level'],
    categorical_mappings={
        'completed': {'Dropped': 0, 'Completed': 1}
    }
)
```

### Example 3: College Admissions
```python
prepare_dataset_template(
    csv_path='admissions.csv', 
    dataset_name='college_admissions',
    feature_columns=['gpa', 'sat_score', 'extracurriculars'],
    target_column='admitted',
    demographic_columns=['gender', 'race', 'income_bracket'],
    categorical_mappings={
        'admitted': {'Rejected': 0, 'Accepted': 1}
    }
)
```

---

## Validation Checklist

After running the preparation:

- [ ] File exists: `data/your_dataset/data_dictionary.pkl`
- [ ] GUI check shows your dataset: `python run_fairness_gui.py --check`
- [ ] Can load in GUI without errors
- [ ] Features are numerical
- [ ] Target is binary (0/1)
- [ ] Demographics are encoded as integers

---

## Common Issues

**"No datasets found"**
- Run from the project root directory
- Check the file path: `data/dataset_name/data_dictionary.pkl`

**"KeyError" or crashes**
- Use the validation: `validate_data_dictionary(your_data_dict)`
- Check that all required columns exist in your CSV

**GUI shows dataset but can't train**
- Ensure features are numerical (not text)
- Check for missing values in your data
- Verify target is binary 0/1

---

## What Happens Next?

Once your dataset is prepared:

1. **GUI Analysis**: Basic bias detection and model training
2. **Advanced Framework**: Use full research pipeline with 50+ fairness algorithms  
3. **Research**: Publish results on algorithmic fairness in education

**Ready to go deeper?** Check out `DATASET_PREPARATION.md` for the complete guide!

---

*Need the example dataset to test?*
```python
from dataset_preparation_guide import example_student_grades
example_student_grades()  # Creates a sample dataset
``` 