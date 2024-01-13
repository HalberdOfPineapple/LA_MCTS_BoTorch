#!/bin/bash

# Navigate to the specified directory
cd /mnt/c/Users/Viktor/Desktop/Cam_2023_2024/Research_Project/botorch_expr/LAMCTS/LA-MCTS-Paper

# Activate the conda environment
conda activate mcts

# Run the python commands with different functions and settings
# python run.py -f ackley -d 20 -n 1000 -s turbo
python run.py -f levy -d 20 -n 1000 -s turbo
# python run.py -f rosenbrock -d 20 -n 1000 -s turbo
# python run.py -f rastrigin -d 20 -n 1000 -s turbo

# python run.py -f ackley -d 50 -n 3000 -s turbo
# python run.py -f levy -d 50 -n 3000 -s turbo
# python run.py -f rosenbrock -d 50 -n 3000 -s turbo
# python run.py -f rastrigin -d 50 -n 3000 -s turbo # TODO