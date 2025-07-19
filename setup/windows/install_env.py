"""
Windows-only script to create a 'debiased' virtual environment using Python 3.11.

powershell -ExecutionPolicy Bypass -File setup.ps1

Requirements:
    - Python 3.11 must be installed and in PATH
    - No support for conda or other OSes
"""

import os
import subprocess
import shutil
import sys


def python_311_exists():
    """Check if python 3.11 is available in PATH"""
    exe = shutil.which("python")
    if not exe:
        return None

    try:
        version_output = subprocess.check_output([exe, "--version"], text=True)
        if version_output.startswith("Python 3.11"):
            return exe
    except subprocess.CalledProcessError:
        pass

    return None


def debiased_venv_exists(path="debiased"):
    """Check if the venv 'debiased' exists in the current directory (Windows layout)"""
    venv_path = os.path.abspath(path)
    return (
        os.path.isdir(venv_path) and
        os.path.exists(os.path.join(venv_path, "Scripts")) and
        os.path.exists(os.path.join(venv_path, "pyvenv.cfg"))
    )


def create_venv(python_exe, path="debiased"):
    """Create the virtual environment"""
    env_path = os.path.abspath(path)
    print(f"📦 Creating virtual environment at: {env_path}")
    try:
        subprocess.run([python_exe, "-m", "venv", env_path], check=True)
        print("✅ Virtual environment created.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        sys.exit(2)


def main():
    python_exe = python_311_exists()
    if not python_exe:
        print("❌ Python 3.11 not found in PATH.")
        print("➡️  Please install Python 3.11 from https://www.python.org/downloads/windows/")
        sys.exit(1)

    if debiased_venv_exists():
        print("✅ Virtual environment 'debiased' already exists.")
    else:
        create_venv(python_exe)


if __name__ == "__main__":
    main()
