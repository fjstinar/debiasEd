# DebiasED Jadouille GUI - Bias Mitigation for Educational Data

A comprehensive graphical user interface for the `debiased_jadouille` package, providing an end-to-end workflow for bias mitigation in educational data analysis.

## Features

### Complete Bias Mitigation Pipeline
- **Data Loading & Exploration**: Load CSV files or use sample datasets
- **Bias Mitigation**: Choose between preprocessing or inprocessing approaches
- **Model Training**: Train standard or fair machine learning models
- **Fairness Evaluation**: Comprehensive fairness metrics and visualizations
- **Results Export**: Save models, results, and reports

### Data Management
- CSV file import with automatic column detection
- Sample educational dataset generator
- Interactive data preview and configuration
- Sensitive attribute selection
- Target variable configuration

### Bias Mitigation Approaches

#### **Preprocessing Methods (15+ Available)**
- **Basic Methods**: Calders Reweighting, SMOTE Oversampling
- **Fair Representation Learning**: Zemel, Lahoti adversarial methods
- **Synthetic Data Generation**: Chakraborty controlled bias synthesis
- **Advanced Fair Sampling**: Dablain, Zelaya oversampling and SMOTE variants
- **Disparate Impact Reduction**: Alabdulmohsin binary debiasing
- **Data Quality**: Li training data debugging, Cock data cleaning
- **Resampling Variants**: Iosifidis attribute/target-based methods

#### **Inprocessing Methods (8+ Available)**
- **Constraint-Based**: Zafar Fair Constraints
- **Adversarial Training**: Chen Multi-Accuracy, Gao Fair Adversarial Networks
- **Deep Learning**: Islam Fairness-Aware Learning
- **Causal Fairness**: Kilbertus Fair Prediction
- **Distribution Matching**: Liu Fair Distribution Matching
- **Gradient-Based**: Grari Gradient-Based Fairness
- **Synthesis During Training**: Chakraborty In-Process Synthesis

### Machine Learning Models
- **Logistic Regression**: With configurable regularization
- **Decision Trees**: With depth and split controls
- **Support Vector Machines**: SVM classification
- **Random Forest**: Ensemble methods

### Evaluation & Fairness Metrics

#### Standard ML Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

#### Fairness Metrics
- Demographic Parity
- Equal Opportunity
- Equalized Odds
- Calibration
- Predictive Parity

### Visualizations
- Data distribution plots
- Preprocessing results visualization
- Confusion matrices
- ROC curves
- Feature importance plots
- Fairness metrics comparisons

## Installation

### Prerequisites
- Python 3.7 or higher
- tkinter (usually included with Python)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Alternative Installation
If you don't have the requirements file, install manually:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

### Quick Start
1. **Launch the GUI**:
   ```bash
   python run_gui.py
   ```
   Or directly:
   ```bash
   python debiased_jadouille_gui.py
   ```

2. **Load Data**:
   - Click "Load CSV File" to import your data
   - Or click "Load Sample Dataset" for a demonstration

3. **Configure Data**:
   - Select your target column
   - Choose sensitive attributes (gender, race, etc.)

4. **Apply Preprocessing (Optional)**:
   - **This step is completely optional!**
   - Choose a preprocessing method if desired
   - Configure method parameters
   - Click "Apply Preprocessing"

5. **Choose Model & Bias Mitigation**:
   - Select a machine learning model
   - Choose bias mitigation approach:
     - **None**: Standard training (works without any preprocessing)
     - **Preprocessed**: Use preprocessed data (only if you applied preprocessing)
     - **Inprocessing**: Fair training algorithm (works with raw data)
   - Configure model and method parameters

6. **Train Model**:
   - Set train/test split
   - Click "Train Model"

6. **Evaluate**:
   - Select evaluation metrics
   - Click "Evaluate Model"
   - View results and visualizations

7. **Export Results**:
   - Save trained models
   - Export results to CSV
   - Generate comprehensive reports

## GUI Interface Overview

### Tab 1: Data
- **Data Loading**: Import CSV files or generate sample data
- **Data Configuration**: Select target and sensitive attributes
- **Data Preview**: Interactive table showing your dataset

### Tab 2: Preprocessing
- **Data Transformation**: Choose from 15+ preprocessing techniques
- **Parameter Configuration**: Tune method-specific parameters
- **Results Visualization**: See before/after/comparison of data changes
- **Optional Step**: Can train models without preprocessing

### Tab 3: Modeling
- **Model Selection**: Choose your machine learning algorithm
- **Bias Mitigation Choice**: None, preprocessed data, or inprocessing
- **Smart Method Filtering**: Only shows inprocessing methods compatible with your model
- **Parameter Tuning**: Configure model hyperparameters
- **Training Controls**: Set test split and train models

### Tab 4: Evaluation
- **Metrics Selection**: Choose ML and fairness metrics
- **Results Display**: View comprehensive evaluation results
- **Visualizations**: Interactive plots and charts

### Tab 5: Results
- **Model Comparison**: Compare different model configurations
- **Export Options**: Save models, results, and reports
- **Configuration Management**: Save and load experimental setups

## Supported Data Formats

### Input Data Requirements
- **Format**: CSV files
- **Structure**: Tabular data with headers
- **Requirements**:
  - At least one target variable
  - At least one sensitive attribute
  - Sufficient samples for train/test split

### Example Data Structure
```csv
student_id,gender,age,previous_grade,study_hours,attendance,socioeconomic_status,pass
1,Male,20,75,5.2,0.85,Medium,1
2,Female,19,82,6.1,0.92,High,1
3,Male,21,68,3.8,0.78,Low,0
...
```

## Bias Mitigation Methods

### **New Workflow: Model-First Bias Mitigation**

#### **Why Model Selection Comes First**
Inprocessing methods are tightly coupled with specific model types since they modify the training process itself. The new workflow ensures you only see compatible options.

#### **Three Bias Mitigation Approaches**

**None (Standard Training)**
- Train your model normally without bias mitigation
- Works directly with raw data - no preprocessing required
- Fastest option, useful for baselines

**Preprocessed Data**
- Apply data transformations before training
- Only available if you applied preprocessing first
- Works with any model type
- Model-agnostic approach

**Inprocessing (Fair Training)**  
- Integrate fairness directly into model training
- Works directly with raw data - no preprocessing required
- Model-specific algorithms
- Strongest fairness guarantees

#### **Smart Compatibility Filtering**
- **Logistic Regression**: Zafar, Kilbertus, Liu, Grari methods
- **Decision Tree**: Chakraborty synthesis methods  
- **SVM**: Zafar, Liu constraint-based methods
- **Random Forest**: Chen, Gao, Islam adversarial methods

### Preprocessing Techniques (15+ Methods)
**Basic Resampling & Reweighting:**
1. **Calders Reweighting**: Adjusts sample weights to reduce bias
2. **SMOTE Oversampling**: Synthetic minority oversampling technique

**Fair Representation Learning:**
3. **Zemel Fair Representations**: Learns fair data representations using prototypes
4. **Lahoti Representation Learning**: Adversarial training for fair representations

**Synthetic Data Generation:**
5. **Chakraborty Synthetic Data**: Generates synthetic fair data with controlled bias

**Advanced Fair Sampling:**
6. **Dablain Fair Over-Sampling**: Holistic approach to bias and imbalance
7. **Zelaya Fair Over-Sampling**: Equal group representation with class balance
8. **Zelaya Fair SMOTE**: SMOTE with fairness constraints

**Disparate Impact Reduction:**
9. **Alabdulmohsin Binary Debiasing**: Reduces disparate impact without demographics

**Data Quality & Debugging:**
10. **Li Training Data Debugging**: Identifies and corrects biased samples
11. **Cock Fair Data Cleaning**: Removes samples contributing to unfairness

**Resampling Variants (Iosifidis Methods):**
12. **Iosifidis Resample Attribute**: Attribute-based resampling
13. **Iosifidis Resample Target**: Target-based resampling
14. **Iosifidis SMOTE Attribute**: SMOTE for attribute balance
15. **Iosifidis SMOTE Target**: SMOTE for target balance

### Inprocessing Techniques (8+ Methods)
**Constraint-Based Fair Learning:**
1. **Zafar Fair Constraints**: Incorporates fairness constraints into optimization
2. **Kilbertus Fair Prediction**: Causal fairness through prediction methods

**Adversarial Training:**
3. **Chen Multi-Accuracy Adversarial Training**: Multiple accuracy objectives
4. **Gao Fair Adversarial Networks**: Adversarial networks for fair representations

**Deep Learning Fairness:**
5. **Islam Fairness-Aware Learning**: Specialized loss functions and regularization

**Advanced Optimization:**
6. **Liu Fair Distribution Matching**: Optimal transport techniques
7. **Grari Gradient-Based Fairness**: Direct fairness metric optimization
8. **Chakraborty In-Process Synthesis**: Real-time synthetic sample generation

### Model Types
1. **Logistic Regression**: Linear classification with regularization
2. **Decision Trees**: Non-linear tree-based classification
3. **SVM**: Support Vector Machine classification
4. **Random Forest**: Ensemble tree-based method

## Fairness Metrics Explained

### Demographic Parity
Ensures equal positive prediction rates across groups.

### Equal Opportunity
Ensures equal true positive rates across groups.

### Equalized Odds
Ensures equal true positive and false positive rates across groups.

### Calibration
Ensures prediction probabilities reflect actual outcomes across groups.

### Predictive Parity
Ensures equal positive predictive values across groups.

## Export Options

### Model Export
- **Format**: Pickle (.pkl) files
- **Includes**: Trained model, encoders, feature names
- **Usage**: Can be loaded for future predictions

### Results Export
- **Format**: CSV files
- **Content**: All metrics for all models and groups
- **Structure**: Tabular format for further analysis

### Configuration Export
- **Format**: JSON files
- **Content**: All experimental settings
- **Usage**: Reproduce experiments with same settings

### Reports
- **Format**: Text files (PDF coming soon)
- **Content**: Comprehensive evaluation summary
- **Includes**: Data summary, methods used, results

## Troubleshooting

### Common Issues

1. **"No data loaded" error**
   - Ensure you've loaded a dataset first
   - Check that the CSV file is properly formatted

2. **"Please select target column" error**
   - Choose a target variable in the Data tab
   - Ensure the column contains binary or categorical values

3. **Import errors**
   - Install required dependencies: `pip install -r requirements.txt`
   - Ensure Python version is 3.7 or higher

4. **GUI not appearing**
   - Check if tkinter is installed: `python -c "import tkinter"`
   - On Linux: `sudo apt-get install python3-tk`

### Performance Tips
- For large datasets (>10,000 rows), consider sampling
- Close visualization windows when not needed
- Use simpler models for initial exploration

## Technical Details

### Architecture
- **Framework**: tkinter with matplotlib integration
- **Data Processing**: pandas and numpy
- **Machine Learning**: scikit-learn
- **Visualization**: matplotlib and seaborn

### File Structure
```
├── debiased_jadouille_gui.py    # Main GUI application
├── run_gui.py                   # Startup script
├── requirements.txt             # Dependencies
├── GUI_README.md               # This documentation
└── debiased_jadouille-0.0.20/  # Original package
    └── src/
        └── debiased_jadouille/
            ├── mitigation/
            ├── predictors/
            └── crossvalidation/
```

## Future Enhancements

### Planned Features
- PDF report generation
- More visualization options
- Advanced parameter tuning
- Batch processing capabilities
- Integration with more bias mitigation methods

### Contributing
To contribute to the GUI development:
1. Test the current functionality
2. Report bugs or suggest features
3. Submit improvements to the codebase

## License

This GUI application is built for the `debiased_jadouille` package, which is licensed under the GNU General Public License v3.0. See the original package LICENSE file for details.

## Support

For issues specific to the GUI:
1. Check this README for common solutions
2. Verify all dependencies are installed
3. Test with the sample dataset first

For issues with the underlying bias mitigation methods, refer to the original `debiased_jadouille` package documentation.

---

**Happy Debiasing!** 