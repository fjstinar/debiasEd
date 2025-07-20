#!/usr/bin/env bash
set -e

echo "Step 1: Installing virtual environment (if needed)..."
python3 setup/unix/install_env.py

echo "Step 2: Activating 'debiased' virtual environment..."
source debiased/bin/activate

echo "Step 3: Installing dependencies..."
pip install --upgrade pip
pip install -r setup/requirements.txt

echo "Step 4: Load your dataset"
read -p "Do you want to use your own dataset? (yes or no) " own_dataset

while [[ "$own_dataset" == "yes" ]]; do
    echo "We are ready to load your own dataset!..."
    read -p "Absolute path to your dataset: " own_path

    echo "Rename the columns in your Excel file as follows:"
    echo "- Features: start with 'feature ' (e.g. 'feature previous grade')"
    echo "- Demographics: start with 'demo'"
    echo "- Label column: must be called 'label'"

    read -p "Name of your dataset: " dataset_name

    python setup/scripts/convert_data.py --path "$own_path" --name "$dataset_name"

    read -p "Do you want to load another dataset? (yes or no) " own_dataset
done

echo "Data Import Done!"
echo "Launching DebiasEd GUI..."
python setup/scripts/run_fairness_gui.py
