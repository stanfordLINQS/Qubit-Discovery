from typing import Any, Optional

import dill as pickle
from SQcircuit import Circuit


def load_record(url: str) -> Any:
    try:
        with open(url, 'rb') as f:
            record = pickle.load(f)
        return record
    except FileNotFoundError:
        return None


# TODO: Generalize codename to account for element ordering
# (ex. for N=4, JJJL and JJLJ should be distinct)
def lookup_codename(num_junctions: int, num_inductors: int) -> Optional[str]:
    if num_inductors == 0 and num_junctions == 2:
        return "JJ"
    if num_inductors == 1 and num_junctions == 1:
        return "JL"
    if num_inductors == 0 and num_junctions == 3:
        return "JJJ"
    if num_inductors == 1 and num_junctions == 2:
        return "JJL"
    if num_inductors == 2 and num_junctions == 1:
        return "JLL"
    return None


def code_to_codename(circuit_code: str) -> str:
    if circuit_code == "JJ":
        return "Transmon"
    if circuit_code == "JL":
        return "Fluxonium"
    return circuit_code


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
    ) #TODO: implement


def load_final_circuit(circuit_record: str) -> Circuit:
    with open(circuit_record, 'rb') as f:
        try:
            while True:
                last_circ = pickle.load(f)
        except EOFError:
            pass
    return last_circ
