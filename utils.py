import logging
import os
import torch 
import numpy as np
from typing import List
from node import Node

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, 'logs')
IBEX_LOG_DIR = os.path.join(BASE_DIR, 'ibex_logs')
PAPER_LOG_DIR = os.path.join(BASE_DIR, 'LA-MCTS-paper', 'logs')
PAPER_IBEX_LOG_DIR = os.path.join(BASE_DIR, 'LA-MCTS-paper', 'ibex_logs')
CONFIG_DIR = os.path.join(BASE_DIR, 'configs')
PATH_DIR = os.path.join(BASE_DIR, 'paths')
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
if not os.path.exists(CONFIG_DIR): os.makedirs(CONFIG_DIR)
if not os.path.exists(PATH_DIR): os.makedirs(PATH_DIR)
if not os.path.exists(IBEX_LOG_DIR): os.makedirs(IBEX_LOG_DIR)
if not os.path.exists(PAPER_LOG_DIR): os.makedirs(PAPER_LOG_DIR)
if not os.path.exists(PAPER_IBEX_LOG_DIR): os.makedirs(PAPER_IBEX_LOG_DIR)

LOGGER: logging.Logger = None
EXPR_NAME: str = None

def init_logger(expr_name: str, in_ibex: bool=False):
    # Setup logging
    print(f"Initialize logger for experiment: {expr_name}")

    log_dir = IBEX_LOG_DIR if in_ibex else LOG_DIR
    log_filename = os.path.join(log_dir, f'{expr_name}.log')
    
    logging.basicConfig(filename=log_filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # Get the logger
    global LOGGER
    LOGGER = logging.getLogger(expr_name)

def set_expr_name(expr_name: str):
    global EXPR_NAME
    EXPR_NAME = expr_name

def get_expr_name():
    global EXPR_NAME
    return EXPR_NAME

def get_logger():
    global LOGGER
    assert LOGGER is not None, "[utils.py] the global ogger has not been initialized. Please call init_logger() first."
    return LOGGER

def print_log(msg: str):
    print(msg)
    global LOGGER
    if LOGGER is not None:
        LOGGER.info(msg)