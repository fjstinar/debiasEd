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
    Create a virtual environment with Python 3.11 named 'debiased'.
    Activation is simulated by launching a subprocess within the environment.
    """
    env_path = os.path.abspath('debiased')

    try:
        print(f"⚖️ Creating virtual environment debiased using python 3.11")
        subprocess.run(["python3.11", "-m", "venv", env_path], check=True)
        print("⚖️ Environment created ⚖️")
    except FileNotFoundError:
        print(f"❌  python 3.11 not found. Make sure Python 3.11 is installed.")
        return
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create environment: {e}")
        return

def check_debiased_exists():
    conda_deb = conda_env_exists()
    conda_env = debiased_env_exists()

    if not conda_env:
        create_and_activate_env()

if __name__ == "__main__":
    check_debiased_exists()
    