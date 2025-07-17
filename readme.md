
## How to run?
conda create -n test python=3.11 -y

conda activate test

pip install -r requirements.txt

## -------------------------------

## DVC Commands

git init

dvc init

dvc repro

dvc dag

dvc metrics show

dvc dag --dot > dvc_pipeline.dot 

brew install graphviz  # macOS
sudo apt install graphviz  # Ubuntu

dot -Tpng dvc_pipeline.dot -o dvc_pipeline.png
