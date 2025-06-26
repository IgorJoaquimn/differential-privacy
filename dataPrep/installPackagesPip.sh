#!/bin/sh

# If you'd rather use conda to install the packages, but don't want to copy a conda environment file,
# you may use this script to install the required packages.

pip install pandas scikit-learn transformers opacus ipykernel numpy && pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118