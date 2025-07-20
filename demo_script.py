#!/usr/bin/env python3
"""
Demonstration script for the DebiasEd GUI

This script shows how to use the GUI components and demonstrates
the complete workflow programmatically.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def create_sample_educational_dataset(n_samples=1000, save_path="sample_educational_data.csv"):
    """
    Create a sample educational dataset with bias for demonstration
    
    Args:
        n_samples (int): Number of samples to generate
        save_path (str): Path to save the CSV file
    
    Returns:
        pd.DataFrame: Generated dataset
    """
    print(f"Creating sample educational dataset with {n_samples} samples...")
    
    np.random.seed(42)
    
    # Generate demographic features with potential bias
    gender = np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4])
    ethnicity = np.random.choice(['White', 'Black', 'Hispanic', 'Asian', 'Other'], 
                                n_samples, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    socioeconomic = np.random.choice(['Low', 'Medium', 'High'], 
                                   n_samples, p=[0.3, 0.5, 0.2])
    
    # Generate educational features
    age = np.random.normal(20, 2, n_samples)
    age = np.clip(age, 18, 25)  # Clip to reasonable range
    
    previous_grade = np.random.normal(75, 10, n_samples)
    previous_grade = np.clip(previous_grade, 50, 100)
    
    study_hours = np.random.exponential(5, n_samples)
    study_hours = np.clip(study_hours, 0, 20)
    
    attendance = np.random.uniform(0.6, 1.0, n_samples)
    
    # Create features that might be correlated with sensitive attributes
    parent_education = []
    for i in range(n_samples):
        if socioeconomic[i] == 'High':
            parent_education.append(np.random.choice(['Bachelor', 'Master', 'PhD'], p=[0.3, 0.4, 0.3]))
        elif socioeconomic[i] == 'Medium':
            parent_education.append(np.random.choice(['High School', 'Bachelor', 'Master'], p=[0.4, 0.5, 0.1]))
        else:
            parent_education.append(np.random.choice(['High School', 'Bachelor'], p=[0.8, 0.2]))
    
    # Create biased target variable (pass/fail)
    # Introduce bias: males and high socioeconomic status have higher pass rates
    pass_prob = 0.5  # Base probability
    
    # Bias factors
    gender_bias = np.where(gender == 'Male', 0.1, -0.1)
    socio_bias = np.where(socioeconomic == 'High', 0.15, 
                         np.where(socioeconomic == 'Medium', 0.05, -0.2))
    
    # Feature-based probability (more realistic)
    feature_effect = (
        (previous_grade - 75) * 0.01 +
        study_hours * 0.02 +
        (attendance - 0.8) * 0.3
    )
    
    # Combine all effects
    final_prob = pass_prob + gender_bias + socio_bias + feature_effect
    final_prob = np.clip(final_prob, 0.1, 0.9)  # Keep probabilities reasonable
    
    pass_outcome = np.random.binomial(1, final_prob)
    
    # Create DataFrame
    data = {
        'student_id': range(1, n_samples + 1),
        'gender': gender,
        'ethnicity': ethnicity,
        'age': np.round(age, 1),
        'socioeconomic_status': socioeconomic,
        'parent_education': parent_education,
        'previous_grade': np.round(previous_grade, 1),
        'study_hours': np.round(study_hours, 1),
        'attendance': np.round(attendance, 2),
        'pass': pass_outcome
    }
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(save_path, index=False)
    print(f"Dataset saved to {save_path}")
    
    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"Features: {len(df.columns) - 1}")
    print(f"Target variable: pass (0/1)")
    print("\nSensitive attributes:")
    print(f"  Gender: {df['gender'].value_counts().to_dict()}")
    print(f"  Ethnicity: {df['ethnicity'].value_counts().to_dict()}")
    print(f"  Socioeconomic: {df['socioeconomic_status'].value_counts().to_dict()}")
    
    print("\nTarget distribution:")
    print(f"  Overall pass rate: {df['pass'].mean():.3f}")
    
    print("\nBias analysis:")
    for gender_val in df['gender'].unique():
        rate = df[df['gender'] == gender_val]['pass'].mean()
        print(f"  {gender_val} pass rate: {rate:.3f}")
    
    for socio_val in df['socioeconomic_status'].unique():
        rate = df[df['socioeconomic_status'] == socio_val]['pass'].mean()
        print(f"  {socio_val} SES pass rate: {rate:.3f}")
    
    return df

def demonstrate_bias_analysis(df):
    """
    Demonstrate bias analysis on the dataset
    
    Args:
        df (pd.DataFrame): Dataset to analyze
    """
    print("\n=== Bias Analysis ===")
    
    # Demographic parity analysis
    print("\nDemographic Parity Analysis:")
    overall_pass_rate = df['pass'].mean()
    print(f"Overall pass rate: {overall_pass_rate:.3f}")
    
    # Gender bias
    print("\nGender bias:")
    for gender in df['gender'].unique():
        gender_df = df[df['gender'] == gender]
        pass_rate = gender_df['pass'].mean()
        bias = pass_rate - overall_pass_rate
        print(f"  {gender}: {pass_rate:.3f} (bias: {bias:+.3f})")
    
    # Socioeconomic bias
    print("\nSocioeconomic bias:")
    for ses in df['socioeconomic_status'].unique():
        ses_df = df[df['socioeconomic_status'] == ses]
        pass_rate = ses_df['pass'].mean()
        bias = pass_rate - overall_pass_rate
        print(f"  {ses}: {pass_rate:.3f} (bias: {bias:+.3f})")
    
    # Intersectional analysis
    print("\nIntersectional analysis (Gender x SES):")
    for gender in df['gender'].unique():
        for ses in df['socioeconomic_status'].unique():
            subset = df[(df['gender'] == gender) & (df['socioeconomic_status'] == ses)]
            if len(subset) > 0:
                pass_rate = subset['pass'].mean()
                count = len(subset)
                print(f"  {gender} x {ses}: {pass_rate:.3f} (n={count})")

def create_gui_usage_guide():
    """Create a step-by-step usage guide for the GUI"""
    
    guide = """
=== DebiasEd GUI Usage Guide ===

This guide walks you through using the GUI application for bias mitigation.

STEP 1: Launch the GUI
----------------------
Option A: Use the startup script
    python run_gui.py

Option B: Launch directly
    python debiased_jadouille_gui.py

STEP 2: Load Data (Data Tab)
-------------------------------
1. Click "Load Sample Dataset" for a quick start
   OR
   Click "Load CSV File" to load your own data
   
2. Select your target column (e.g., "pass")
3. Check boxes for sensitive attributes (e.g., "gender", "socioeconomic_status")
4. Review the data preview table

STEP 3: Apply Preprocessing (Preprocessing Tab)
--------------------------------------------------
1. Choose a bias mitigation method:
   - Calders Reweighting (recommended for beginners)
   - Zemel Learning Fair Representations
   - Chakraborty Synthetic Data
   - SMOTE Oversampling
   - Fair Oversampling

2. Adjust method parameters if needed
3. Click "Apply Preprocessing"
4. Review the preprocessing results visualization

STEP 4: Train Model (Modeling Tab)
-------------------------------------
1. Select a machine learning model:
   - Logistic Regression (good for interpretability)
   - Decision Tree (non-linear, interpretable)
   - SVM (good for complex boundaries)
   - Random Forest (ensemble method)

2. Configure model parameters
3. Set the test split ratio (default: 20%)
4. Click "Train Model"
5. Wait for training to complete

STEP 5: Evaluate Model (Evaluation Tab)
------------------------------------------
1. Select evaluation metrics:
   ML Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
   Fairness Metrics: Demographic Parity, Equal Opportunity, etc.

2. Click "Evaluate Model"
3. Review the metrics in the summary table
4. Examine the visualizations:
   - Confusion matrix
   - ROC curve
   - Feature importance
   - Prediction distributions

5. Click "Generate Report" for a comprehensive summary

STEP 6: Export Results (Results Tab)
---------------------------------------
1. View model comparison table
2. Export options:
   - "Export Model": Save trained model (.pkl)
   - "Export Results (CSV)": Save all metrics
   - "Export Report (PDF)": Comprehensive report
   
3. Save/Load configurations for reproducibility

TIPS FOR BEST RESULTS:
----------------------
- Start with the sample dataset to learn the interface
- Try different preprocessing methods and compare results
- Look for fairness-accuracy trade-offs in the evaluation
- Save your configuration before experimenting
- Use the execution log to track your progress

INTERPRETING FAIRNESS METRICS:
------------------------------
- Demographic Parity: Equal positive prediction rates across groups
- Equal Opportunity: Equal true positive rates across groups
- Values closer to 0 indicate better fairness
- Perfect fairness (0) may come at the cost of accuracy

TROUBLESHOOTING:
---------------
- If GUI doesn't appear: Check tkinter installation
- If imports fail: Install requirements.txt
- If evaluation fails: Ensure preprocessing was applied first
- For large datasets: Consider using a sample first
"""
    
    print(guide)
    
    # Save guide to file
    with open("GUI_Usage_Guide.txt", "w") as f:
        f.write(guide)
    print("\nUsage guide saved to 'GUI_Usage_Guide.txt'")

def main():
    """Main demonstration function"""
    print("=" * 60)
    print("DebiasEd GUI - Demonstration Script")
    print("=" * 60)
    
    # Create sample dataset
    print("\n1. Creating sample educational dataset...")
    df = create_sample_educational_dataset()
    
    # Analyze bias in the dataset
    print("\n2. Analyzing bias in the dataset...")
    demonstrate_bias_analysis(df)
    
    # Create usage guide
    print("\n3. Creating GUI usage guide...")
    create_gui_usage_guide()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nFiles created:")
    print("  - sample_educational_data.csv (sample dataset)")
    print("  - GUI_Usage_Guide.txt (step-by-step guide)")
    print("\nNext steps:")
    print("  1. Run: python run_gui.py")
    print("  2. Load the sample dataset")
    print("  3. Follow the usage guide")
    print("  4. Experiment with different methods!")
    print("\nHappy debiasing!")

if __name__ == "__main__":
    main() 