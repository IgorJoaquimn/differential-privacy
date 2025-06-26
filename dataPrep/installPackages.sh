#!/bin/sh

# If you'd rather use conda to install the packages, but don't want to copy a conda environment file,
# you may use this script to install the required packages.

conda install pandas scikit-learn pytorch torchvision torchaudio pytorch-cuda=11.8 transformers opacus ipykernel numpy faiss-gpu -c pytorch -c nvidia -c huggingface -c conda-forge -y