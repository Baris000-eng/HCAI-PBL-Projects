# Environment Setup Guide

This guide walks you through setting up a virtual environment and installing the required packages for **Project3** (which contains the Django and ML-based Python codebase, e.g., `ml_backbone.py`).

Choose **one** of the methods below based on your operating system and package manager preference.

---

## Method 1: Using `venv` (Standard Python)

### MacOS / Linux
Open your terminal and run the following commands:

```bash
# 1. Navigate to the project folder
cd path_to_the_project3

# 2. Create the virtual environment folder named 'hcai_env'
python3 -m venv hcai_env

# 3. Activate the environment
source hcai_env/bin/activate

# 4. Navigate to the project3 directory
cd project3

# 5. Install the required dependencies
pip install -r requirements.txt

# 6. Deactivate the environment when you are done
deactivate
```

### Windows
Open your Command Prompt or PowerShell and run:

:: 1. Navigate to the project folder
cd path_to_the_project3

:: 2. Create the virtual environment folder named 'hcai_env'
python -m venv hcai_env

:: 3. Activate the environment
.\hcai_env\Scripts\activate

:: 4. Navigate to the project3 directory
cd project3

:: 5. Install the required dependencies
pip install -r requirements.txt

:: 6. Deactivate the environment when you are done
deactivate

## Method 2: Using conda (Anaconda / Miniconda)
- Works across all operating systems (Windows, MacOS, Linux).

# 1. Create the environment named 'hcai_env' with Python 3.12
conda create --name hcai_env python=3.12 -y

# 2. Activate the environment
conda activate hcai_env

# 3. Navigate to the project3 directory
cd path_to_the_project/project3

# 4. Install the required dependencies
# Option A: Using pip (Recommended for mixed Django/ML environments)
pip install -r requirements.txt

# Option B: Using conda
conda install --file requirements.txt

# 5. Deactivate the environment when you are done
conda deactivate


Note on Project Structure:
The project3 directory is located at the same level as the project2 folder. Ensure you are inside project3 
before running the dependency installation commands so that pip can locate the requirements.txt file.