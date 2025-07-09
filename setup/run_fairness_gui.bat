@echo off
REM DebiasEd Fairness GUI - Windows Launcher
REM =========================================

echo.
echo ============================================================
echo   DebiasEd: Fairness in Educational Machine Learning  
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo.
    echo Please install Python 3.7+ from: https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

REM Run the GUI
echo Starting DebiasEd GUI...
echo.
python run_fairness_gui.py

REM Keep window open if there was an error
if %errorlevel% neq 0 (
    echo.
    echo Press any key to close...
    pause >nul
) 