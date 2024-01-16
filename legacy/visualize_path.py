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
def plot_decision_boundaries(classifiers, labels,  range_values, iteration_number, save_path):
    xx, yy = np.meshgrid(np.linspace(range_values[0], range_values[1], 500),
                         np.linspace(range_values[0], range_values[1], 500))

    plt.figure(figsize=(10, 10))
    plane = np.ones(xx.shape)
    for clf, label in zip(classifiers, labels):
        try:
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]) == label
        except:
            continue
        # print(Z)
        Z = Z.reshape(xx.shape)
        plane = np.logical_and(plane, Z)
    plt.contour(xx, yy, plane, levels=[0], linewidths=2, colors='black')

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
    cls_files = [f for f in os.listdir(cls_dir) if f.endswith('.pkl')]
    label_files = [f for f in os.listdir(cls_dir) if f.endswith('.txt')]

    # Partition file names by iteration number
    partitioned_files = {}
    partitioned_label_files = {}
    for cls_file, label_file in zip(cls_files, label_files):
        iteration_number = int(cls_file.split('_')[0])
        if iteration_number not in partitioned_files:
            partitioned_files[iteration_number] = []
            partitioned_label_files[iteration_number] = []
        
        partitioned_files[iteration_number].append(cls_file)
        partitioned_label_files[iteration_number].append(label_file)

    classifiers_by_path = {}
    labels_by_path = {}
    for iteration_number, files in partitioned_files.items():
        classifiers_by_path[iteration_number] = [
            load_classifier(os.path.join(cls_dir, file_name)) for file_name in files
        ]

        labels_by_path[iteration_number] = []
        for label_file in partitioned_label_files[iteration_number]:
            with open(os.path.join(cls_dir, label_file), 'r') as file:
                labels_by_path[iteration_number].append(int(file.read()))

    
    # Define the range for the plot
    range_values = (-10, 10)
    for iteration_number in classifiers_by_path.keys():
        # Plot decision boundaries
        classifiers, labels = classifiers_by_path[iteration_number], labels_by_path[iteration_number]
        plot_decision_boundaries(classifiers, labels, range_values, iteration_number, cls_dir)

if __name__ == '__main__':
    expr_name = 'mcts_ackley2d_200'
    main(expr_name)