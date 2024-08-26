import os
import logging
import sys
from typing import Any

import dill as pickle
from matplotlib import pyplot as plt
from SQcircuit import Circuit

def add_stdout_to_logger(
        logger: logging.Logger,
        level: int =logging.INFO
) -> logging.Handler:
    """Add a handler to ``logger`` to print to stdout.
    """
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)

    return handler


def load_record(url: str) -> Any:
    if not os.path.exists(url):
        print(f"The file '{url}' does not exist.")
        return None

    try:
        with open(url, 'rb') as f:
            record = pickle.load(f)
        return record

    except EOFError:
        print(f"Warning: The file '{url}' is empty or corrupted.")
        return None


def load_initial_circuit(circuit_record: str) -> Circuit:
    with open(circuit_record, 'rb') as f:
        first_circ = pickle.load(f)
    return first_circ


def load_final_circuit(circuit_record: str) -> Circuit:
    with open(circuit_record, 'rb') as f:
        try:
            while True:
                last_circ = pickle.load(f)
        except EOFError:
            pass
    return last_circ


def load_all_circuits(circuit_record: str) -> Circuit:
    circuits = []
    with open(circuit_record, 'rb') as f:
        try:
            while True:
                next_circuit = pickle.load(f)
                circuits += [next_circuit, ]
        except EOFError:
            pass
    return circuits


def set_plotting_defaults(single_color=False):
    plt.rcParams['lines.linewidth'] = 2.2
    plt.rcParams["xtick.major.size"] = 5
    plt.rcParams["xtick.major.width"] = 1.2
    plt.rcParams["ytick.major.size"] = 5
    plt.rcParams["ytick.major.width"] = 1.2
    plt.rcParams["axes.titlesize"] = 24
    plt.rcParams["font.size"] = 24
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    color_list = ['#8D1514', '#007662', '#333131', '#D1C295', '#FD7C34', '#6B3B6D']
    if single_color:
        color_list = ['#8D1514', ]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
