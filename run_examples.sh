#!/bin/bash

python pipelines.py --model tem # Train the TEM model

python pipelines.py --model tem --eval <checkpoint_path> # Test the TEM model with a checkpoint

python pipelines.py --model madlib --eval <checkpoint_path> # Test the MADlib model with a checkpoint

python pipelines.py --run_private # Train the baseline model with DP-SGD