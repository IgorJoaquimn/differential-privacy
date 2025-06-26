#!/bin/bash

# python pipeline.py --target_epsilon 12.5 # Train and test the baseline model
# python pipeline.py --target_epsilon 12.5 --run_private # Train the baseline model with DP-SGD
# python pipeline.py --folder results1 --target_epsilon 5000 --model madlib # Train and test the MADlib model
# python pipeline.py --folder results1 --target_epsilon 500 --model madlib # Train and test the MADlib model

# python pipeline.py --folder results1 --target_epsilon 12.5 --model tem # Train and test the TEM model

# python pipeline.py --folder results1 --target_epsilon 5 --run_private # Train the baseline model with DP-SGD
# python pipeline.py --folder results1 --target_epsilon 5 --model madlib # Train and test the MADlib model
# python pipeline.py --folder results1  --target_epsilon 5 --model tem # Train and test the TEM model

# python pipeline.py --folder results1 --target_epsilon 50 --run_private # Train the baseline model with DP-SGD
# python pipeline.py --folder results1 --target_epsilon 50 --model madlib # Train and test the MADlib model
# python pipeline.py --folder results1  --target_epsilon 50 --model tem # Train and test the TEM model

# python pipeline.py --folder results1 --target_epsilon 100 --run_private # Train the baseline model with DP-SGD
# python pipeline.py --folder results1 --target_epsilon 100 --model madlib # Train and test the MADlib model
# python pipeline.py --folder results1  --target_epsilon 100 --model tem # Train and test the TEM model

python pipeline.py --folder results1 --target_epsilon 5 --model madlib --run_private # Train the MADlib model with DP-SGD
python pipeline.py --folder results1 --target_epsilon 12.5 --model madlib --run_private # Train the MADlib model with DP-SGD
python pipeline.py --folder results1 --target_epsilon 50 --model madlib --run_private # Train the MADlib model with DP-SGD
python pipeline.py --folder results1 --target_epsilon 100 --model madlib --run_private # Train the MADlib model with DP-SGD

python pipeline.py --folder results1 --target_epsilon 5 --model tem --run_private # Train the TEM model with DP-SGD
python pipeline.py --folder results1 --target_epsilon 12.5 --model tem --run_private # Train the TEM model with DP-SGD
python pipeline.py --folder results1 --target_epsilon 50 --model tem --run_private # Train the TEM model with DP-SGD
python pipeline.py --folder results1 --target_epsilon 100 --model tem --run_private # Train the TEM model with DP-SGD

# python pipeline.py --target_epsilon 30 --run_private # Train the baseline model with DP-SGD
# python pipeline.py  --target_epsilon 30 --model tem # Train and test the TEM model
# python pipeline.py --target_epsilon 30 --model tem --run_private # Train the TEM model with DP-SGD

# python pipeline.py --target_epsilon 75 --run_private # Train the baseline model with DP-SGD
# python pipeline.py  --target_epsilon 75 --model tem # Train and test the TEM model
# python pipeline.py --target_epsilon 75 --model tem --run_private # Train the TEM model with DP-SGD

# python pipeline.py --target_epsilon 500 --run_private # Train the baseline model with DP-SGD
# python pipeline.py  --target_epsilon 500 --model tem # Train and test the TEM model
# python pipeline.py --target_epsilon 500 --model tem --run_private # Train the TEM model with DP-SGD