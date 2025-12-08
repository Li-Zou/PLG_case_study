# Build & Track ML Pipelines 

## How to run?

conda create -n test python=3.11 -y

conda activate test

pip install -r requirements.txt

git init

python -m src.run_all --config configs/default.yaml





#Alternative method for running the code.

# Build & Track ML Pipelines with DVC

## How to run?

conda create -n test python=3.11 -y

conda activate test

pip install -r requirements.txt

## DVC Commands

git init

dvc init

dvc repro


