#!/usr/bin/env bash
set -e

echo "============================================================"
echo "DebiasEd - Unix/Linux Setup and Launch Script"
echo "============================================================"
echo ""

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not installed."
    echo "Please install Python 3 using your package manager:"
    echo "   Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-venv python3-pip"
    echo "   CentOS/RHEL: sudo yum install python3 python3-venv python3-pip"
    echo "   Fedora: sudo dnf install python3 python3-venv python3-pip"
    exit 1
fi

echo "FOUND: Python 3 found: $(python3 --version)"

# Check if pip is available
if ! command -v pip3 &> /dev/null; then
    echo "ERROR: pip3 is required but not found."
    echo "Please install pip3 using your package manager."
    exit 1
fi

echo "FOUND: pip3 found: $(pip3 --version)"

# Check if virtual environment exists, create if not
if [ ! -d "debiased_env" ]; then
    echo "SETUP: Creating virtual environment..."
    python3 -m venv debiased_env
fi

echo "SETUP: Activating virtual environment..."
source debiased_env/bin/activate

echo "SETUP: Installing/updating dependencies..."
pip install --upgrade pip
pip install -r setup/general/requirements.txt

echo ""
echo "DATA: Dataset Setup"
read -p "Do you want to convert your own dataset? (yes or no) " own_dataset

while [[ "$own_dataset" == "yes" ]]; do
    echo ""
    echo "DATA: Dataset Conversion Setup:"
    echo "We'll help you convert your dataset to the correct format!"
    echo ""
    
    read -p "INPUT: Absolute path to your CSV file: " own_path
    
    # Check if file exists
    if [ ! -f "$own_path" ]; then
        echo "ERROR: File not found: $own_path"
        echo "Please check the path and try again."
        continue
    fi
    
    echo ""
    echo "INFO: IMPORTANT: Your CSV file must have columns named as follows:"
    echo "   - Features: start with 'feature ' (e.g., 'feature previous_grade')"
    echo "   - Demographics: start with 'demo ' (e.g., 'demo gender')"
    echo "   - Target/Label: must be named exactly 'label'"
    echo ""
    echo "Example column names:"
    echo "   feature age, feature grade, demo gender, demo ethnicity, label"
    echo ""
    
    read -p "INPUT: Name for your converted dataset: " dataset_name
    
    # Validate dataset name (no spaces, no special characters)
    if [[ ! "$dataset_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        echo "ERROR: Dataset name should only contain letters, numbers, underscores, and hyphens."
        continue
    fi
    
    echo ""
    echo "PROCESSING: Converting dataset..."
    python3 setup/general/convert_data.py --path "$own_path" --name "$dataset_name"
    
    if [ $? -eq 0 ]; then
        echo "SUCCESS: Dataset converted successfully!"
        echo "         Saved to: data/$dataset_name/data_dictionary.pkl"
    else
        echo "ERROR: Dataset conversion failed. Please check your file format."
    fi
    
    echo ""
    read -p "Do you want to convert another dataset? (yes or no) " own_dataset
done

echo ""
echo "LAUNCH: Starting DebiasEd GUI..."
echo "============================================================"
python3 gui/run_gui.py
