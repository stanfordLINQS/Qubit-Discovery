from typing import Any

import dill as pickle
from matplotlib import pyplot as plt
from SQcircuit import Circuit


def load_record(url: str) -> Any:
    try:
        with open(url, 'rb') as f:
            record = pickle.load(f)
        return record

    except EOFError:
        print(f"Warning: The file '{url}' is empty or corrupted.")
        return None


def add_file_args(parser):
    parser.add_argument(
        '-c', '--codes', type=str, required=True,
        help="Circuit codes to plot, each with <num_runs>"
    )
    parser.add_argument(
        '-o', '--optimization_type', type=str, required=True,
        help="Optimization type to plot"
    )
    parser.add_argument(
        '-n', '--name', required=True,
        help="Name of run, either explict or passed in via YAML."
    )
    parser.add_argument(
        '-s', '--save_circuits', action='store_true',
        help="Unimplemented"
    )  # TODO: implement


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
    plt.rcParams["axes.titlesize"] = 18
    plt.rcParams["font.size"] = 18
    plt.rcParams['xtick.labelsize'] = 15
    plt.rcParams['ytick.labelsize'] = 15
    color_list = ['#8D1514', '#007662', '#333131', '#D1C295', '#FD7C34']
    if single_color:
        color_list = ['#8D1514', ]
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=color_list)
