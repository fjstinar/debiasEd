Write-Host " Creating virtual environment..."
python setup/scripts/install_env.py

Write-Host " Activating virtual environment..."
. .\debiased\Scripts\Activate.ps1

Write-Host " Installing dependencies..."
pip install --upgrade pip
pip install -r setup/requirements.txt

$useDataset = Read-Host "Do you want to use your own dataset? (yes or no)"
while ($useDataset -eq "yes") {
    $own_path = Read-Host "Give the absolute path to your data"
    $dataset_name = Read-Host "What is the name of your dataset?"

    python setup/scripts/convert_data.py --path "$own_path" --name "$dataset_name"

    $useDataset = Read-Host "Do you want to load another dataset? (yes or no)"
}

Write-Host "Data import done."
python setup/scripts/run_fairness_gui.py
