# Build & Track ML Pipelines 

# Build & Track ML Pipelines with DVC

# How to run?

conda create -n test python=3.11 -y

conda activate test

pip install -r requirements.txt

## DVC Commands

git init

dvc init

dvc repro


# Alternative method for running the code.

# How to run?

conda create -n test python=3.11 -y

conda activate test

pip install -r requirements.txt

git init
python -m src.run_all --config configs/default.yaml

