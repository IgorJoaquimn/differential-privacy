#!/bin/bash

python pipelines.py --model tem # Train and test the TEM model

python pipelines.py --model tem --eval checkpoints/tem.pt # Only test the TEM model with a checkpoint

python pipelines.py --model madlib --eval checkpoints/madlib.pt # Test the MADlib model with a checkpoint

python pipelines.py --run_private # Train the baseline model with DP-SGD

python pipelines.py --run_private --eval checkpoints/baseline_private.pt # Test the baseline model with DP-SGD and a checkpoint