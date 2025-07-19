#!/usr/bin/env python3
"""
Creates Virtual Environment to Self Contain Debiase
==============================================

Requirements:
    - Python 3.7+
    - Required packages (will be checked and reported)
    - MAC OS
"""

import os
import subprocess
import shutil
import sys


def conda_env_exists(name="debiased"):
    """Checks if a Python environment on Conda exists
    """
    try:
        result = subprocess.run(
            ["conda", "env", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        return any(line.startswith(name) for line in result.stdout.splitlines())

    except Exception:
        return False

def debiased_env_exists(name='debiased', path="."):
    """
    Check if a Python virtual environment named 'debiased' exists at the given path, for non conda environments
    """
    venv_path = os.path.join(path, name)
    expected_dirs = ["bin", "lib", "pyvenv.cfg"]  # Unix/Mac

    if not os.path.isdir(venv_path):
        return False

    contents = os.listdir(venv_path)
    return all(any(item.startswith(name) for item in contents) for name in expected_dirs)


def create_and_activate_env():
    """
    Create a 'debiased' virtual environment using system-wide python3.11.
    Provides platform-specific guidance if python3.11 is missing.
    """
    env_path = os.path.abspath('debiased')
    python_exe = shutil.which("python3.11")

    if not python_exe:
        print("❌ Python 3.11 not found in PATH.")
        print("💡 To fix this:")

        if sys.platform == "darwin":
            print("   Please download and install Python 3.11 from the official website:")
            print("   https://www.python.org/downloads/macos/")
        elif sys.platform.startswith("linux"):
            print("   Install via your package manager. For example:")
            print("     Debian/Ubuntu: sudo apt update && sudo apt install python3.11 python3.11-venv")
            print("     Fedora:         sudo dnf install python3.11 python3.11-venv")
            print("     Arch:           sudo pacman -S python311")
        else:
            print("   Please install Python 3.11 appropriate for your platform.")

        sys.exit(1)

    print(f"📦 Creating virtual environment at {env_path} using {python_exe}")
    try:
        subprocess.run([python_exe, "-m", "venv", env_path], check=True)
        print("✅ Virtual environment created.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        print("💡 Ensure that the 'venv' module is available in your Python 3.11 installation.")
        sys.exit(2)


def check_debiased_exists():
    conda_deb = conda_env_exists()
    conda_env = debiased_env_exists()

    if not conda_env:
        create_and_activate_env()

if __name__ == "__main__":
    check_debiased_exists()
    