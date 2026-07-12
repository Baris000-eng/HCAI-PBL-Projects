This is the repository including Django-Based Human-Centric Artificial Intelligence Projects. 

# How to Get Started with Development? 

## Firstly, it is recommended to create a virtual environment. 

# Virtual Environment Setup & Dependency Installation

This guide walks you through setting up an isolated Python environment and installing all required libraries for **Project1** (which houses the Django application and ML-based scripts, such as `views.py`). Choose **one** of the configuration paths below depending on your operating system and environment manager preference.

## Method 1: Using `venv` (Standard Python)

### MacOS / Linux
Open your terminal and execute the following commands sequentially:
```bash
# 1. Navigate to your root project directory
cd path_to_the_project

# 2. Initialize a virtual environment named 'hcai_env'
python3 -m venv hcai_env

# 3. Activate the virtual environment
source hcai_env/bin/activate

# 4. Move into the project directory containing the source code
cd project_folder

# 5. Install the required project dependencies
pip install -r requirements.txt

# 6. Deactivate the environment once your session is complete
deactivate```

Windows OS
Open Command Prompt or PowerShell and execute the following commands sequentially:

```bash
1. Navigate to your root project directory
cd path_to_the_project

2. Initialize a virtual environment named 'hcai_env'
python -m venv hcai_env

3. Activate the virtual environment
.\hcai_env\Scripts\activate

4. Move into the 'project1' directory containing the source code
cd project1

5. Install the required project dependencies
pip install -r requirements.txt

6. Deactivate the environment once your session is complete
deactivate
```

## Method 2: Using conda (Anaconda / Miniconda)

This method is platform-independent and works across Windows, MacOS, and Linux. Open your terminal or Conda Prompt and execute the following commands sequentially:

```bash
# 1. Create a new conda environment named 'hcai_env' using Python 3.12 
conda create --name hcai_env python=3.12 -y

# 2. Activate the newly created environment
conda activate hcai_env

# 3. Move into the 'project1' directory containing the source code
cd path_to_the_project/project1

# 4. Install the required project dependencies
# Option A: Standard Pip installation (Recommended for mixed Django/ML environments)
pip install -r requirements.txt
# Option B: Native Conda installation
conda install --file requirements.txt

# 5. Deactivate the environment when you are done working
conda deactivate
```


## Used Library Versions Guide

### Used Python Version: Python 3.12.2

### Used Django Version (django): 5.2.7 

### How to Install? 
```bash
pip install django==5.2.7
```

### Used Scikit-Learn Version (scikit-learn): 1.7.2

### How to Install?
```bash
pip install scikit-learn==1.7.2
```

### Used Datasets Library Version (datasets): 4.2.0

### How to Install?
```bash
pip install datasets==4.2.0
```

### Used Matplotlib Library Version (matplotlib): 3.10.9

### How to Install?
```bash
pip install matplotlib==3.10.9
```


### Used Palmerpenguins Library Version (palmerpenguins): 0.1.6

### How to Install?
```bash
pip install matplotlib==3.10.9
```


### Used Graphviz Library Version (graphviz): 0.21

### How to Install?
```bash
pip install graphviz==0.21
```


### Used Pandas Library Version (pandas): 2.3.3

### How to Install?
```bash
pip install pandas==2.3.3
```


### Used Bottleneck Library Version (bottleneck): 1.6.0

### How to Install?
```bash
pip install bottleneck==1.6.0
```



## Step 1: How to fork the GitHub repository for own usage? 

1. Open your browser and go to the GitHub repository to be forked: https://github.com/ppaamm/HCAI-PBL
2. In the top-right corner of the page, which is at the left of "Watch" button and right of "Star" button, click the "Fork" button.
3. Select your account as the owner, write a repository name of your choice, and optionally write a repository description. 


## Step 2: How to clone forked repository ? 

Download your personal copy of the repository from GitHub to your local machine:

```bash
git clone https://github.com/Baris000-eng/HCAI-PBL.git
```

## Step 3: Navigate to the project folder
Move into the newly created project directory:

```bash
cd HCAI-PBL
```

## Step 4: Verify Current Remote Connections
Check which online repositories your local project is currently linked to. At this stage, it should only show the personal fork, which is origin: 

```bash
git remote -v
```


## Step 5: Link the Original and Upstream Repository to the Local Project
Connect local project to the original author's repository. This allows you to pull down any future updates they make:
```bash
git remote add upstream https://github.com/ppaamm/HCAI-PBL.git
```



## Step 6: Verify the Last Setup
Run the check again. You should now see both origin, which points out to your GitHub account, and upstream, which points out to the original project. 

```bash
git remote -v
```

