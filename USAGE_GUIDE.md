# DebiasEd - Quick Start Guide

## Launching

### **For macOS Users:**
```bash
bash setup/mac/launch_macos.sh
```

### **For Linux/Unix Users:**
```bash
bash setup/linux/setup_unix.sh
```

### **For Windows Users:**
```powershell
powershell -ExecutionPolicy Bypass -File setup/windows/setup_windows.ps1
```

### **Direct GUI Launch** (after setup):
```bash
python gui/run_gui.py
```

## 📁 Repository Structure

```
debiasEd/
├── setup/           # Setup scripts and dependencies
│   ├── mac/         # macOS-specific setup
│   ├── linux/       # Linux/Unix setup  
│   ├── windows/     # Windows setup
│   └── general/     # Common files (requirements.txt, convert_data.py)
├── src/             # Source code (debiased_jadouille package)
├── gui/             # GUI application files
├── data/            # Sample datasets and converted data
├── gui_readme/      # GUI documentation
└── package_readme/  # Package documentation
```


The setup scripts are bash/PowerShell scripts, not Python scripts!

## What Each Setup Script Does

1. **Check Python installation** and version
2. **Create virtual environment** (if needed)
3. **Install dependencies** from `setup/general/requirements.txt`
4. **Offer dataset conversion** using `setup/general/convert_data.py`
5. **Launch the GUI** from `gui/run_gui.py`

## Need Help?

- Check `gui_readme/` for GUI usage instructions
- Check `package_readme/` for package documentation  
- Check `setup/general/SETUP_README.md` for detailed setup information 