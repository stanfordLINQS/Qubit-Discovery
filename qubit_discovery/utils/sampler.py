from collections import defaultdict
import random
from typing import Dict, Iterable, List, Set, Tuple, Union

from scipy.stats import loguniform
import numpy as np

from SQcircuit.circuit import Circuit
from SQcircuit.elements import Capacitor, Element, Inductor, Junction, Loop
from SQcircuit.settings import get_optim_mode

def test_equivalent(code_1: str, code_2: str) -> bool:
    '''Given two strings indicating a sequence of junctions and inductors 
    (ex. JLJ and JJL), determines whether the patterns are equivalent up to 
    cyclic permutations.'''
    assert len(code_1) == len(code_2)
    for i in range(1, len(code_2)):
        if code_1[i:] + code_1[:i] == code_2:
            return True
    return False

def filter(element_codes: Iterable[str]) -> Set[str]:
    '''Given a sequence of junctions and inductors encoded in string format 
    (ex. JJLJ), removes any that are equivalent up to cyclic permutation.'''
    element_codes_list = list(element_codes)
    for i in range(len(element_codes_list)):
        j = i + 1
        while j < len(element_codes_list):
            if test_equivalent(element_codes_list[i], element_codes_list[j]):
                del element_codes_list[j]
                j -= 1
            j += 1
    return set(element_codes_list)

def _generate_topologies(num_elements: int) -> Set[str]:
    '''Generates the set of all unique orderings of N junctions and inductors 
    on a one-loop ring, for later use in sampling random circuit topologies.'''
    assert num_elements >= 1

    element_codes = set()
    element_codes.add('J')  # There should always be at least one junction
    length = 1
    while length < num_elements:
        new_element_codes = set()
        for element_code in element_codes:
            for inductive_element in ['J', 'L']:
                new_element_codes.add(element_code + inductive_element)
        element_codes = new_element_codes
        length += 1
    element_codes = filter(element_codes)
    return element_codes

class CircuitSampler:
    """Class used to randomly sample different circuit configurations."""

    def __init__(
            self,
            num_elements: int,
            capacitor_range: list = [12e-15, 12e-9],
            inductor_range: list = [12e-9, 12e-6],
            junction_range: list = [1e9, 10e9]
    ) -> None:
        self.num_elements = num_elements
        self.topologies = list(_generate_topologies(num_elements))
        self.topologies.sort()
        self.capacitor_range = capacitor_range
        self.inductor_range = inductor_range
        self.junction_range = junction_range
        self.trunc_num = 40

    def sample_circuit(self) -> str:
        sampled_topology = random.sample(self.topologies, 1)[0]
        return sampled_topology
        # Build circuit, assign random values by sampling from range for each element

    def sample_circuit_code(self, codename: str, default_flux=0.5) -> Circuit:
        loop = Loop()
        loop.set_flux(default_flux)
        circuit_elements: Dict[Tuple[int, int], List[Element]] = defaultdict(list)

        element: Union[Junction, Inductor, Capacitor]

        # Edge case: Single junction
        if codename == 'J':
            junction_value = loguniform.rvs(*self.junction_range, size=1)[0]
            junction_value /= (2 * np.pi)
            junction = Junction(junction_value, 'Hz', requires_grad=get_optim_mode(),
                                   min_value=self.junction_range[0], max_value=self.junction_range[1])
            cap_value = loguniform.rvs(*self.capacitor_range, size=1)[0]
            capacitor = Capacitor(cap_value, 'F', requires_grad=get_optim_mode(),
                                min_value=self.capacitor_range[0], max_value=self.capacitor_range[1])
            circuit_elements[(0, 1)] = [junction, capacitor]
            return Circuit(circuit_elements, flux_dist='junctions')

        if 'C' in codename:
            loops = []
        else:
            loops = [loop, ]
        # Add inductive elements to circuit
        for element_idx, element_code in enumerate(codename):
            if element_code == 'J':
                # Add requires grad to element here?
                junction_value = loguniform.rvs(*self.junction_range, size=1)[0]
                junction_value /= (2 * np.pi)
                element = Junction(junction_value, 'Hz', loops=loops, requires_grad=get_optim_mode(),
                                   min_value=self.junction_range[0], max_value=self.junction_range[1])
            elif element_code == 'L':
                # TODO: Include default quality factor Q in inductor?
                inductor_value = loguniform.rvs(*self.inductor_range, size=1)[0]
                element = Inductor(inductor_value, 'H', loops=loops, requires_grad=get_optim_mode(),
                                   min_value=self.inductor_range[0], max_value=self.inductor_range[1])
            elif element_code == 'C':
                cap_value = loguniform.rvs(*self.capacitor_range, size=1)[0]
                element = Capacitor(cap_value, 'F', requires_grad=get_optim_mode(),
                                   min_value=self.capacitor_range[0], max_value=self.capacitor_range[1])

            min_idx = min(element_idx, (element_idx + 1) % len(codename))
            max_idx = max(element_idx, (element_idx + 1) % len(codename))
            if self.num_elements == 2:
                # Edge case for n=2: Two elements on same edge
                circuit_elements[(min_idx, max_idx)] += [element, ]
            else:
                circuit_elements[(min_idx, max_idx)] = [element, ]

        # Introduce all-to-all capacitive coupling
        for first_element_idx in range(len(codename)):
            for second_element_idx in range(first_element_idx + 1, len(codename)):
                capacitor_value = loguniform.rvs(*self.capacitor_range, size=1)[0]
                capacitor = Capacitor(capacitor_value, 'F', requires_grad=get_optim_mode(),
                                      min_value=self.capacitor_range[0], max_value=self.capacitor_range[1])
                circuit_elements[(first_element_idx, second_element_idx)] += [capacitor, ]

        circuit = Circuit(circuit_elements, flux_dist='junctions')
        # If mode j > 100 * mode i, set mode j trunc num to 1
        # circuit.set_trunc_nums([np.pow(1000,1/n), np.pow(1000,1/n), np.pow(1000,1/n), np.pow(1000,1/n)])
        # Weight based on natural frequency?
        return circuit

    def sample_one_loop_circuits(self, 
                                 n: int, 
                                 with_replacement = True
                                 ) -> List[Circuit]:
        circuits = []
        if not with_replacement:
            assert n <= len(self.topologies), "Number of circuit topologies sampled without replacement must be less" \
                                              "than or equal to number of distinct arrangements of inductive elements."
            sampled_topologies = random.sample(self.topologies, n)
        else:
            sampled_topologies = random.choices(self.topologies, k = n)

        for topology in sampled_topologies:
            circuit = self.sample_circuit_code(topology)
            circuits.append(circuit)

        return circuits