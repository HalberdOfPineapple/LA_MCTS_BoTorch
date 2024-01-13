import os 
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC

from utils import PATH_DIR

def load_classifier(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)

# Function to plot decision boundaries for a list of classifiers
def plot_decision_boundaries(classifiers, range_values, iteration_number, save_path):
    xx, yy = np.meshgrid(np.linspace(range_values[0], range_values[1], 100),
                         np.linspace(range_values[0], range_values[1], 100))

    plt.figure(figsize=(10, 10))
    for clf in classifiers:
        try:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        except:
            continue
        print(Z)
        Z = Z.reshape(xx.shape)
        plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

    plt.xlim(range_values)
    plt.ylim(range_values)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Boundaries at Iteration {iteration_number}')

    plt.savefig(os.path.join(save_path, f"vis_{iteration_number}.png"))
    plt.close()

def main(expr_name):
    # Assuming the classifiers are saved in the 'classifiers' folder in the current working directory
    cls_dir = os.path.join(PATH_DIR, expr_name)

    # Get all .pkl files in the directory
    all_files = [f for f in os.listdir(cls_dir) if f.endswith('.pkl')]

    # Partition file names by iteration number
    partitioned_files = {}
    for file_name in all_files:
        iteration_number = int(file_name.split('_')[0])
        if iteration_number not in partitioned_files:
            partitioned_files[iteration_number] = []
        partitioned_files[iteration_number].append(file_name)

    classifiers_by_path = {}
    for iteration_number, files in partitioned_files.items():
        classifiers_by_path[iteration_number] = [
            load_classifier(os.path.join(cls_dir, file_name)) for file_name in files
        ]
    
    # Define the range for the plot
    range_values = (-10, 10)
    for iteration_number, classifiers in classifiers_by_path.items():
        # Plot decision boundaries
        plot_decision_boundaries(classifiers, range_values, iteration_number, cls_dir)

if __name__ == '__main__':
    expr_name = 'mcts_ackley2d_200'
    main(expr_name)