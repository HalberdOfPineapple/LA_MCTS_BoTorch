import logging
import os
import torch 
import numpy as np
from typing import List
from node import Node

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
CONFIG_DIR = os.path.join(BASE_DIR, 'configs')
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
if not os.path.exists(CONFIG_DIR): os.makedirs(CONFIG_DIR)

LOGGER: logging.Logger = None

def init_logger(expr_name: str):
    # Setup logging
    print(f"Initialize logger for experiment: {expr_name}")
    log_filename = os.path.join(LOG_DIR, f'{expr_name}.log')
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Get the logger
    global LOGGER
    LOGGER = logging.getLogger(expr_name)

def get_logger():
    global LOGGER
    assert LOGGER is not None, "[utils.py] the global ogger has not been initialized. Please call init_logger() first."
    return LOGGER


def path_filter(path: List[Node], candidates: torch.Tensor) -> np.array:
    """
    Args:
        path: List[Node] - A list of nodes from the root to the current node
        candidates: torch.Tensor - (num_candidates, dimension)

    Returns:
        choices: np.array - (num_candidates, ) - A boolean array indicating whether each candidate is accepted
    """
    choices: np.array = np.full((candidates.shape[0], ), True)
    for i in range(len(path) - 1):
        curr_node: Node = path[i]
        target_label: int = path[i + 1].label

        labels = curr_node.predict_label(candidates[choices])
        choices[choices] = labels == target_label
        # choices[choices] = labels == (1 - target_label)

        if choices.sum() == 0:
            break

    return choices

def check_path(path: List[Node]):
    leaf: Node = path[-1]
    leaf_X: torch.Tensor = leaf.sample_bag[0]

    in_regions = path_filter(path, leaf_X)
    num_in_regions = in_regions.sum()

    print('-' * 20)
    print(f"[utils.check_path] {num_in_regions} / {leaf_X.shape[0]} samples of the leaf node {leaf.id}"
          f" are in the region of the leaf node {leaf.id} with label {leaf.label}")

    # for i in range(len(path) - 1):
    #     curr_node: Node = path[i]
    #     next_node: Node = path[i+1]
    #     if not next_node in curr_node.children:
    #         raise ValueError(f"Path is not valid from node {curr_node.id} with samples {curr_node.sample_bag[0].shape[0]}"
    #                         f" to node {next_node.id} with samples {next_node.sample_bag[0].shape[0]}. "
    #                         f"Because the next node is not a child of the current node")

    #     target_child_label: int = next_node.label
    #     target_child_data: torch.Tensor = next_node.sample_bag[0] # (num_samples, dimension)
    #     labels = curr_node.classifier.predict(target_child_data) # (num_samples, )
            
    #     # labels of the target child should be predicted to the target child label by the current node's classifier
    #     if not (labels == target_child_label).all():
    #         print(f"Current node's classifier's pos label: {curr_node.classifier.pos_label}")
    #         raise ValueError(f"Path is not valid from node {curr_node.id} with samples {curr_node.sample_bag[0].shape[0]}"
    #                          f" to node {next_node.id} with samples {next_node.sample_bag[0].shape[0]}. "
    #                          f"Because the child should be classified to label {target_child_label} but got labels: {labels}")
