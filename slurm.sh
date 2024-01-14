#!/bin/bash
#SBATCH --job-name=botorch_expr
#SBATCH --output=slurm_logs/job_output_%j.txt
#SBATCH --error=slurm_logs/job_error_%j.txt
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:v100:1

# Activate the conda environment
source /home/${USER}/.bashrc
conda activate mcts

# List of python launch.py commands
# python launch.py -n 200 -e ackley2d -i
# python LA-MCTS-paper/run.py -f ackley -d 2 -n 200 -s turbo -i

python launch.py -n 3000 -e ackley50d -i
python launch.py -n 3000 -e levy50d -i
python launch.py -n 3000 -e rosenbrock50d -i
python launch.py -n 3000 -e rastrigin50d -i

python launch.py -n 3000 -e ackley50d -m turbo -i
python launch.py -n 3000 -e levy50d -m turbo -i 
python launch.py -n 3000 -e rosenbrock50d -m turbo -i
python launch.py -n 3000 -e rastrigin50d -m turbo -i

python LA-MCTS-paper/run.py -f ackley -d 50 -n 3000 -s turbo -i
python LA-MCTS-paper/run.py -f levy -d 50 -n 3000 -s turbo -i
python LA-MCTS-paper/run.py -f rosenbrock -d 50 -n 3000 -s turbo -i
python LA-MCTS-paper/run.py -f rastrigin -d 50 -n 3000 -s turbo -i

# Deactivate conda environment
conda deactivate