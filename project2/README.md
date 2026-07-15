# Environment Setup Guide

This guide walks you through setting up a virtual environment and installing the required packages for **Project2**:

Choose **one** of the methods below based on your operating system and package manager preference.

---

## Method 1: Using `venv` (Standard Python)

### MacOS / Linux
Open your terminal and run the following commands:

```bash
# 1. Create the virtual environment folder named 'hcai_env'
python3 -m venv hcai_env

# 2. Activate the environment
source hcai_env/bin/activate

# 3. Navigate to the project2 directory
cd project2

# 4. Install the required dependencies
pip install -r requirements.txt

# 5. Deactivate the environment when you are done
deactivate
```

### Windows
Open your Command Prompt or PowerShell and run:

```bash
# 1. Create the virtual environment folder named 'hcai_env'
python3 -m venv hcai_env

# 2. Activate the environment
.\hcai_env\Scripts\activate

# 3. Navigate to the project2 directory
cd project2

# 4. Install the required dependencies
pip install -r requirements.txt

# 5. Deactivate the environment when you are done
deactivate
```

## Method 2: Using conda (Anaconda / Miniconda)
- Works across all operating systems (Windows, MacOS, Linux).

```bash
# 1. Create the environment named 'hcai_env' with Python 3.12
conda create --name hcai_env python=3.12 -y

# 2. Activate the environment
conda activate hcai_env

# 3. Navigate to the project2 directory
cd project2

# 4. Install the required dependencies
# Option A: Using pip (Recommended for mixed Django/ML environments)
pip install -r requirements.txt

# Option B: Using conda
conda install --file requirements.txt

# 5. Deactivate the environment when you are done
conda deactivate
```


Note on Project Structure:
The project2 directory is located at the same level as the project1 folder. Ensure you are inside project2 
before running the dependency installation commands so that pip can locate/find the requirements.txt file.

# Project Testing Guide
In order to test the project2, please run the commands below: 

# 1. Navigate to the "HCAI-PBL-PROJECTS" folder
```bash
cd HCAI-PBL-PROJECTS
```
# 2. Run the command below to start testing 
```bash
python manage.py test 
```