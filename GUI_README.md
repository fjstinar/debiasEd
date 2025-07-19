# DebiasEd GUI - Standalone Fairness Tool

A user-friendly graphical interface for bias detection and mitigation in educational machine learning.

## Quick Start

### Option 1: One-Click Launch (Recommended)
```bash
python run_fairness_gui.py
```

This script will:
- Check your Python version
- Install missing packages automatically
- Verify data availability
- Launch the GUI

### Option 2: Manual Setup

1. **Install dependencies:**
   ```bash
   pip install -r gui_requirements.txt
   ```

2. **Run the GUI directly:**
   ```bash
   python gui/unfairness_mitigation_gui.py
   ```

## System Requirements

- **Python 3.11+** (Python 3.11 recommended)
- **Operating System:** Windows, macOS, or Linux
- **Memory:** 2GB RAM minimum
- **Storage:** 500MB for datasets

## What the GUI Does

### Core Features:
- **Load Educational Datasets** (EEDI, OULAD, Student Performance, etc.)
- **Apply Preprocessing Techniques** for bias mitigation (SMOTE, Rebalancing, Reweighting)
- **Train ML Models** with and without preprocessing
- **Compare Fairness Results** side-by-side (Baseline vs Preprocessed)
- **Explore Dataset Features** in an interactive table
- **Analyze Performance Trade-offs** between accuracy and fairness

### Supported Datasets:
- **EEDI**: Educational assessment platform data
- **OULAD**: Open University Learning Analytics
- **XuetangX**: Chinese MOOC platform data
- **Student Performance**: Portuguese school data (Math & Language)
- **Cyprus**: Higher education performance data

## How to Use

1. **Launch the GUI:**
   ```bash
   python run_fairness_gui.py
   ```

2. **Load a Dataset:**
   - Click on any "Load [Dataset]" button
   - Wait for the data to load
   - Explore features in the popup window

3. **Select Preprocessing Method:**
   - Choose from dropdown: None (Baseline), SMOTE, Rebalancing, or Calders
   - Read the description of each method
   - Understand what bias mitigation approach you're applying

4. **Train with Preprocessing:**
   - Click "Train Decision Tree with Preprocessing"
   - System trains both baseline and preprocessed models
   - View comparative results side-by-side

5. **Interpret Results:**
   - Compare baseline vs preprocessed performance
   - Look for improvements (green ↑) or degradations (red ↓)
   - Analyze data size changes from preprocessing
   - Check fairness analysis summary and recommendations

## Troubleshooting

### Common Issues:

**"No module named 'tkinter'"**
- **Ubuntu/Debian:** `sudo apt-get install python3-tk`
- **CentOS/RHEL:** `sudo yum install tkinter`
- **macOS:** Use Python from python.org (not Homebrew)

**"No datasets found"**
- Make sure you're in the project root directory
- Data files should be in: `data/[dataset]/data_dictionary.pkl`
- Run data preparation notebooks first if needed

**Package installation fails**
- Try: `pip install --user -r gui_requirements.txt`
- Or use conda: `conda install numpy pandas scikit-learn matplotlib`

### System Check:
```bash
python run_fairness_gui.py --check
```

## Getting Help

**Command line help:**
```bash
python run_fairness_gui.py --help
```

**Features:**
- System diagnostics
- Package installation
- Data verification
- Error reporting

## Preprocessing Methods Available

### **None (Baseline)**
- No preprocessing applied
- Train on original data
- Useful for comparison baseline

### **SMOTE (Oversampling)**
- **Technique:** Synthetic Minority Oversampling Technique
- **Purpose:** Creates synthetic examples to balance classes
- **Best for:** Datasets with class imbalance
- **Reference:** Chawla et al. (2002)

### **Rebalancing**
- **Technique:** Rebalances demographic groups through sampling
- **Purpose:** Ensures equal representation across groups
- **Best for:** Datasets with demographic imbalance
- **Effect:** Changes dataset size through resampling

### **Calders (Reweighting)**
- **Technique:** Reweights instances for demographic independence
- **Purpose:** Adjusts sample importance to achieve fairness
- **Best for:** General bias mitigation without changing data
- **Reference:** Calders et al. (2009)

## For Researchers

This GUI provides a simplified interface to the full DebiasEd framework. For advanced features:

- **Full pipeline:** See `src/script_classification.py`
- **All algorithms:** Browse `src/mitigation/` (30+ preprocessing methods available)
- **Configuration:** Edit `src/configs/`
- **Custom experiments:** Use the CLI tools

## Sample Workflow

### **Basic Fairness Analysis:**
1. **Start with EEDI dataset** (good test case)
2. **Load and explore** the dataset features
3. **Train baseline** (No preprocessing) to see initial performance
4. **Apply SMOTE preprocessing** and compare results
5. **Try different preprocessing** methods (Rebalancing, Calders)
6. **Analyze trade-offs** between fairness and accuracy

### **Advanced Research:**
1. **Use multiple datasets** to test generalizability
2. **Document preprocessing effects** on different demographic groups
3. **Compare preprocessing methods** systematically
4. **Move to full framework** for 30+ algorithms and research-grade analysis

---

**Happy bias hunting!** 