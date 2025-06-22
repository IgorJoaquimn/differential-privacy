#!/bin/bash

python pipelines.py --model tem # Train the TEM model

python pipelines.py --model tem --test <checkpoint_path> # Test the TEM model with a checkpoint

python pipelines.py --model madlib --test <checkpoint_path> # Test the MADlib model with a checkpoint

python pipelines.py --run_private # Train the baseline model with DP-SGD