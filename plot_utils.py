from typing import Optional, Tuple

from SQcircuit import Capacitor, Circuit, Inductor, Junction

from qubit_discovery.optimization.utils import flatten

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

def get_optimal_key(loss_record, code: Optional[str]=None):
  optimal_loss = 1e100
  optimal_key = None

  for circuit, circuit_code, l in loss_record.keys():
    key = (circuit, circuit_code, 'total_loss')
    if len(loss_record[key]) == 0:
        continue
    loss = loss_record[key][-1]
    if loss < optimal_loss and (code in key or code is None):
        optimal_loss = loss
        optimal_key = key

  return optimal_key

def get_element_counts(circuit: Circuit) -> Tuple[int, int, int]:
    """Gets counts of each type of circuit element."""
    inductor_count = sum([type(xi) is Inductor for xi in
                          flatten(list(circuit.elements.values()))])
    junction_count = sum([type(xi) is Junction for xi in
                          flatten(list(circuit.elements.values()))])
    capacitor_count = sum([type(xi) is Capacitor for xi in
                           flatten(list(circuit.elements.values()))])
    return junction_count, inductor_count, capacitor_count