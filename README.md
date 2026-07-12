This is the repository including Django-Based Human-Centric Artificial Intelligence Projects. 

# How to Get Started with Development? 

## Used Library Versions Guide

### Used Python Version: Python 3.12.2
### Used Django Version (django): 5.2.7
### Used Scikit-Learn Version (scikit-learn): 1.7.2
### Used Datasets Library Version (datasets): 4.2.0
### Used Matplotlib Library Version (matplotlib): 3.10.9
### Used Palmerpenguins Library Version (palmerpenguins): 0.1.6
### Used Graphviz Library Version (graphviz): 0.21
### Used Pandas Library Version (pandas): 2.3.3
### Used Bottleneck Library Version (bottleneck): 1.6.0



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

