# To install the packages needed, these steps can be followed: 

# Create a virtual environment using conda or venv.

### venv virtual environment activation and deactivation in a MacOS / Linux OS ###
# Navigate to your project folder
cd path_to_the_project

# Create the virtual environment folder, named 'hcai_env'
python3 (or python) -m venv hcai_env

# Activate the environment
source hcai_env/bin/activate

# Go to the folder named "project3", which is at the same level with the folder named "project2" and 
# which contains django and ML-based Python codes (e.g. ml_backbone.py). 

# To install all libraries needed by Project1 in the virtual environment 'hcai_env', we need to run the following command: 
pip install -r requirements.txt 

# Deactivate when you are done
deactivate
### venv virtual environment activation and deactivation in a MacOS / Linux OS ###


### venv virtual environment activation and deactivation in a Windows OS ###
# Navigate to your project folder
cd path_to_the_project

# Create the virtual environment folder
python -m venv hcai_env

# Activate the environment
.hcai_env\Scripts\activate

# Go to the folder named "project3", which is at the same level with the folder named "project2" and 
# which contains django and ML-based Python codes (e.g. ml_backbone.py). 

# To install all libraries needed by Project3 in the virtual environment 'hcai_env', we need to run the following command: 
pip install -r requirements.txt 

# Deactivate when you are done
deactivate
### venv virtual environment activation and deactivation in a Windows OS ###


### conda #####

# Create the environment named hcai_env
conda create --name hcai_env python=3.12

# Activate the environment
conda activate myenv

# Go to the folder named "project3", which is at the same level with the folder named "project2" and 
# which contains django and ML-based Python codes (e.g. ml_backbone.py). 

# To install all libraries needed by Project3 in the virtual environment 'hcai_env', we need to run the following command: 
pip install -r requirements.txt OR conda install --file requirements.txt

# Deactivate when you are done
conda deactivate

### conda #####



