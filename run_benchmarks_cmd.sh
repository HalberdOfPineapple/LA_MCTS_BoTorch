
# Activate the conda environment
source /home/${USER}/.bashrc
conda activate mcts


python launch.py -n 1000 -e ackley20d -m gp_ei
python launch.py -n 1000 -e levy20d -m gp_ei
python launch.py -n 1000 -e rosenbrock20d -m gp_ei
python launch.py -n 1000 -e rastrigin20d -m gp_ei

# python launch.py -n 3000 -e ackley50d -i
# python launch.py -n 3000 -e levy50d -i
# python launch.py -n 3000 -e rosenbrock50d -i
# python launch.py -n 3000 -e rastrigin50d -i

# python launch.py -n 3000 -e ackley50d -m turbo -i
# python launch.py -n 3000 -e levy50d -m turbo -i 
# python launch.py -n 3000 -e rosenbrock50d -m turbo -i
# python launch.py -n 3000 -e rastrigin50d -m turbo -i

# python LA-MCTS-paper/run.py -f ackley -d 50 -n 3000 -s turbo -i
# python LA-MCTS-paper/run.py -f levy -d 50 -n 3000 -s turbo -i
# python LA-MCTS-paper/run.py -f rosenbrock -d 50 -n 3000 -s turbo -i
# python LA-MCTS-paper/run.py -f rastrigin -d 50 -n 3000 -s turbo -i

# Deactivate conda environment
conda deactivate