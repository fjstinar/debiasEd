# DebiasEd - Windows Setup and Launch Script
# To run this script:
# powershell -ExecutionPolicy Bypass -File setup_windows.ps1

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "DebiasEd - Windows Setup and Launch Script" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Check if Python 3 is installed
try {
    $pythonVersion = python --version 2>$null
    if ($pythonVersion) {
        Write-Host "FOUND: Python found: $pythonVersion" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "ERROR: Python 3 is required but not installed." -ForegroundColor Red
    Write-Host "Please install Python 3 from https://www.python.org/" -ForegroundColor Yellow
    Write-Host "Make sure to check 'Add Python to PATH' during installation." -ForegroundColor Yellow
    exit 1
}

# Check if virtual environment exists, create if not
if (-not (Test-Path "debiased_env")) {
    Write-Host "SETUP: Creating virtual environment..." -ForegroundColor Yellow
    python -m venv debiased_env
}

Write-Host "SETUP: Activating virtual environment..." -ForegroundColor Yellow
& .\debiased_env\Scripts\Activate.ps1

Write-Host "SETUP: Installing/updating dependencies..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

Write-Host ""
Write-Host "DATA: Dataset Setup" -ForegroundColor Cyan
$useDataset = Read-Host "Do you want to convert your own dataset? (yes or no)"

while ($useDataset -eq "yes") {
    Write-Host ""
    Write-Host "DATA: Dataset Conversion Setup:" -ForegroundColor Cyan
    Write-Host "We'll help you convert your dataset to the correct format!" -ForegroundColor Green
    Write-Host ""
    
    $own_path = Read-Host "INPUT: Absolute path to your CSV file"
    
    # Check if file exists
    if (-not (Test-Path $own_path)) {
        Write-Host "ERROR: File not found: $own_path" -ForegroundColor Red
        Write-Host "Please check the path and try again." -ForegroundColor Yellow
        continue
    }
    
    Write-Host ""
    Write-Host "INFO: IMPORTANT: Your CSV file must have columns named as follows:" -ForegroundColor Yellow
    Write-Host "   - Features: start with 'feature ' (e.g., 'feature previous_grade')" -ForegroundColor White
    Write-Host "   - Demographics: start with 'demo ' (e.g., 'demo gender')" -ForegroundColor White
    Write-Host "   - Target/Label: must be named exactly 'label'" -ForegroundColor White
    Write-Host ""
    Write-Host "Example column names:" -ForegroundColor Cyan
    Write-Host "   feature age, feature grade, demo gender, demo ethnicity, label" -ForegroundColor Gray
    Write-Host ""
    
    $dataset_name = Read-Host "INPUT: Name for your converted dataset"
    
    # Validate dataset name (no spaces, no special characters)
    if ($dataset_name -notmatch "^[a-zA-Z0-9_-]+$") {
        Write-Host "ERROR: Dataset name should only contain letters, numbers, underscores, and hyphens." -ForegroundColor Red
        continue
    }
    
    Write-Host ""
    Write-Host "PROCESSING: Converting dataset..." -ForegroundColor Yellow
    python convert_data.py --path "$own_path" --name "$dataset_name"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "SUCCESS: Dataset converted successfully!" -ForegroundColor Green
        Write-Host "         Saved to: data\$dataset_name\data_dictionary.pkl" -ForegroundColor Gray
    } else {
        Write-Host "ERROR: Dataset conversion failed. Please check your file format." -ForegroundColor Red
    }
    
    Write-Host ""
    $useDataset = Read-Host "Do you want to convert another dataset? (yes or no)"
}

Write-Host ""
Write-Host "LAUNCH: Starting DebiasEd GUI..." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Cyan
python run_gui.py
