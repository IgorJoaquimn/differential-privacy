#!/bin/bash

# python pipeline.py --target_epsilon 5 # Train and test the baseline model
python pipeline.py  --target_epsilon 5 --model tem # Train and test the TEM model
python pipeline.py --target_epsilon 5 --model madlib # Train and test the MADlib model
python pipeline.py --target_epsilon 5 --run_private # Train the baseline model with DP-SGD
python pipeline.py --target_epsilon 5 --model tem --run_private # Train the TEM model with DP-SGD
python pipeline.py --target_epsilon 5 --model madlib --run_private # Train the MADlib model with DP-SGD

# python pipeline.py --target_epsilon 7.5 # Train and test the baseline model
python pipeline.py  --target_epsilon 7.5 --model tem # Train and test the TEM model
python pipeline.py --target_epsilon 7.5 --model madlib # Train and test the MADlib model
python pipeline.py --target_epsilon 7.5 --run_private # Train the baseline model with DP-SGD
python pipeline.py --target_epsilon 7.5 --model tem --run_private # Train the TEM model with DP-SGD
python pipeline.py --target_epsilon 7.5 --model madlib --run_private # Train the MADlib model with DP-SGD

# python pipeline.py --target_epsilon 10 # Train and test the baseline model
# python pipeline.py  --target_epsilon 10 --model tem # Train and test the TEM model
# python pipeline.py --target_epsilon 10 --model madlib # Train and test the MADlib model
# python pipeline.py --target_epsilon 10 --run_private # Train the baseline model with DP-SGD
# python pipeline.py --target_epsilon 10 --model tem --run_private # Train the TEM model with DP-SGD
# python pipeline.py --target_epsilon 10 --model madlib --run_private # Train the MADlib model with DP-SGD

# python pipeline.py --target_epsilon 12.5 # Train and test the baseline model
# python pipeline.py  --target_epsilon 12.5 --model tem # Train and test the TEM model
# python pipeline.py --target_epsilon 12.5 --model madlib # Train and test the MADlib model
# python pipeline.py --target_epsilon 12.5 --run_private # Train the baseline model with DP-SGD
# python pipeline.py --target_epsilon 12.5 --model tem --run_private # Train the TEM model with DP-SGD
# python pipeline.py --target_epsilon 12.5 --model madlib --run_private # Train the MADlib model with DP-SGD