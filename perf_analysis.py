import os
import numpy as np
import matplotlib.pyplot as plt
from utils import LOG_DIR


def read_data(log_path):
    function_values = []
    with open(log_path, 'r') as file:
        for line in file:
            # Split the line into function value and coordinates, then extract the function value
            func_val = line.split('],')[0].strip()
            func_val = func_val.replace('[', '').replace(']', '')
            function_values.append(float(func_val))
    return np.array(function_values)

def plot_trend(log_name, function_values):
    # plt.plot(function_values)
    plt.plot(np.minimum.accumulate(function_values))

    plt.title(log_name)
    plt.xlabel('Iterations')
    plt.ylabel('Function Value')

    plt.savefig(os.path.join(LOG_DIR, f"{log_name}_plot.png"))

def main(log_name):
    log_file_path = os.path.join(LOG_DIR, f"{log_name}.log")
    function_values = -read_data(log_file_path)

    plot_trend(log_name, function_values)


if __name__ == "__main__":
    main('mcts_ackley20d_1000_data')