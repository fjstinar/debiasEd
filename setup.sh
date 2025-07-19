#!/bin/bash
set -e

# Resolve project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR" && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="debiased"

echo "📦 Step 1: Installing virtual environment (if needed)..."
python3 setup/scripts/install_env.py

echo "🐍 Step 2: Activating virtual environment..."
. $ENV_NAME/bin/activate

echo "⬆️ Step 3: Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r setup/requirements.txt

echo "🔍 Step 4: Checking tkinter availability..."
if ! python -c "import tkinter" &>/dev/null; then
  echo "❌ tkinter is not installed in this Python build."
  echo "💡 On Ubuntu/Debian, run: sudo apt install python3-tk"
  exit 1
fi

# Only on Linux: check if X11/GUI is available
if [[ "$OSTYPE" == "linux-gnu"* ]] && ! xset q &>/dev/null; then
  echo "❌ No GUI display found (X11 not available). GUI won't launch."
  echo "💡 Are you running in a headless terminal or SSH session?"
  exit 1
fi

echo "📁 Step 5: Do you want to use your own dataset? (yes or no)"
read own_dataset

while [[ "$own_dataset" == "yes" ]]; do
  echo "📌 Provide the absolute path to your Excel file:"
  read own_path

  echo -e "\n📝 Rename your Excel columns as follows:"
  echo "- Features → start with 'feature '"
  echo "- Demographics → start with 'demo'"
  echo "- Label column → named 'label'"

  echo "🧾 Enter a name for your dataset (no spaces):"
  read dataset_name

  python setup/scripts/convert_data.py --path "$own_path" --name "$dataset_name"

  echo -e "\n🔁 Load another dataset? (yes or no)"
  read own_dataset
done

echo "📊 Step 6: Data import done. Launching GUI..."
python setup/scripts/run_fairness_gui.py
