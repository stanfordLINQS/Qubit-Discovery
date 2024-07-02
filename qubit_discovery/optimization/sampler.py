from collections import defaultdict
from typing import List, Union

from scipy.stats import loguniform
import numpy as np

import SQcircuit as sq

from SQcircuit.circuit import Circuit
from SQcircuit.elements import Capacitor, Inductor, Junction


def find_last_element_index_outside_parentheses(circuit_code: str) -> int:
    """Find last element index outside parentheses. This is the last element
    that encloses the main loop.

    Parameters
    ----------
        circuit_code:
            A string specifying the circuit code.
    """
    stack = []
    last_index = -1

    for i, char in enumerate(circuit_code):
        if char == '(':
            stack.append(char)
        elif char == ')':
            if stack:
                stack.pop()
        else:
            if not stack:
                last_index = i

    return last_index


class CircuitSampler:
    """Class used to randomly sample different circuit configurations.

    Parameters
    ----------
        capacitor_range:
            A list specifying the lower bound and upper bound for the
            capacitors.
        inductor_range:
            A list specifying the lower bound and upper bound for the inductors.
        junction_range:
            A list specifying the lower bound and upper bound for the junctions.
    """

    def __init__(
            self,
            capacitor_range: List[float],
            inductor_range: List[float],
            junction_range: List[float],
    ) -> None:

        self.capacitor_range = capacitor_range
        self.inductor_range = inductor_range
        self.junction_range = junction_range
        self.loop = sq.Loop(0.5)

        self.special_circuit_codes = [
            "transmon",
            "flux_qubit",
        ]

    def get_elem(
        self,
        elem_str: str,
        main_loop: bool,
    ) -> Union[Capacitor, Inductor, Junction]:
        """Get a random capacitor, inductor, or junction based on the specified
        range for sampler.

        Parameters
        ----------
            elem_str:
                A string element type that should be either "L", "C", or "J".
            main_loop:
                A boolean that indicates whether we are in the main loop of
                the circuit or not.
        """

        loops = []
        if main_loop:
            loops = [self.loop,]

        if elem_str == "J":
            junc_value = loguniform.rvs(*self.junction_range, size=1)[0]
            junc_value /= (2 * np.pi)
            elem = sq.Junction(
                junc_value,
                'Hz',
                loops=loops,
                requires_grad=sq.get_optim_mode(),
                min_value=self.junction_range[0],
                max_value=self.junction_range[1]
            )
        elif elem_str == "L":
            ind_value = loguniform.rvs(*self.inductor_range, size=1)[0]
            elem = sq.Inductor(
                ind_value,
                'H',
                loops=loops,
                requires_grad=sq.get_optim_mode(),
                min_value=self.inductor_range[0],
                max_value=self.inductor_range[1]
            )
        elif elem_str == "C":
            cap_value = loguniform.rvs(*self.capacitor_range, size=1)[0]
            elem = Capacitor(
                cap_value,
                'F',
                requires_grad=sq.get_optim_mode(),
                min_value=self.capacitor_range[0],
                max_value=self.capacitor_range[1]
            )
        else:
            raise ValueError("elem_str should be either 'J', 'L', or 'C' ")

        return elem

    def sample_special_circuit(self, circuit_code):

        if circuit_code == "transmon":
            junc_1 = self.get_elem('J', main_loop=True)
            junc_2 = junc_1

            cap = self.get_elem('C', main_loop=False)

            elements = {(0, 1): [junc_1, junc_2, cap]}

        elif circuit_code == "flux_qubit":
            junc_1 = self.get_elem('J', main_loop=True)
            junc_2 = junc_1
            junc_3 = self.get_elem('J', main_loop=True)

            cap_1 = self.get_elem('C', main_loop=False)
            cap_2 = cap_1
            cap_3 = self.get_elem('C', main_loop=False)

            elements = {
                (0, 1): [junc_1, cap_1],
                (1, 2): [junc_2, cap_2],
                (2, 0): [junc_3, cap_3]
            }

        else:
            raise ValueError("The circuit code is not supported!")

        return sq.Circuit(elements, flux_dist='junctions')

    def add_elem_to_elements(
        self,
        elem_str: str,
        node_1: int,
        node_2: int,
        elements: dict,
        main_loop: bool = False,
    ) -> None:
        """Add a random element to a specific edge key of elements within the
        specified range of element values.

        Parameters
        ----------
            elem_str:
                A string element type that should be either "L", "C", or "J".
            node_1:
                An integer specifying node 1 of the edge.
            node_2:
                An integer specifying node 2 of the edge.
            elements:
                A dictionary that contains the circuit's elements at each branch
                of the circuit.
            main_loop:
                A boolean that indicates whether we are in the main loop of
                the circuit or not.
        """

        elem = self.get_elem(elem_str, main_loop)

        if (node_2, node_1) in elements:
            elements[(node_2, node_1)].append(elem)
        else:
            elements[(node_1, node_2)].append(elem)

    @staticmethod
    def if_capacitor_between_nodes(
        i: int,
        j: int,
        elements: dict
    ) -> bool:
        """Check whether there is a capacitor between nodes i and j.

        Parameters
        ----------
            i:
                An integer specifying node i of the branch.
            j:
                An integer specifying node 2 of the edge.
            elements:
                A dictionary that contains the circuit's elements at each branch
                of the circuit.
        """

        if (i, j) in elements:
            if any(isinstance(elem, Capacitor) for elem in elements[(i, j)]):
                return True
        if (j, i) in elements:
            if any(isinstance(elem, Capacitor) for elem in elements[(j, i)]):
                return True

        return False

    def add_all_to_all_cap_to_elements(
        self,
        n_nodes: int,
        elements: dict,
    ) -> None:
        """Add capacitors to all branches of the circuit.

        Parameters
        ----------
            n_nodes:
                An Integer specifying the number of nodes in the circuit.
            elements:
                A dictionary that contains the circuit's elements at each branch
                of the circuit.
        """

        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                # we prevent more than one capacitor on each branch
                if self.if_capacitor_between_nodes(i, j, elements):
                    continue

                self.add_elem_to_elements(
                    elem_str="C",
                    node_1=i,
                    node_2=j,
                    elements=elements,
                )

    def sample_circuit_code(self, circuit_code) -> Circuit:
        """Sample a random circuit with specified circuit code.

        Parameters
        ----------
            circuit_code:
                A string specifying the circuit code.
        """

        if circuit_code in self.special_circuit_codes:
            return self.sample_special_circuit(circuit_code)

        current_node = 0
        hold_nodes = []
        idx = 0
        elements = defaultdict(list)

        final_i = find_last_element_index_outside_parentheses(circuit_code)

        for i in range(len(circuit_code)):

            if circuit_code[i] in ("J", "L", "C"):

                if i == final_i and i == len(circuit_code)-1:
                    self.add_elem_to_elements(
                        elem_str=circuit_code[i],
                        node_1=current_node,
                        node_2=0,
                        elements=elements,
                        main_loop=not bool(len(hold_nodes)),
                    )
                    continue

                if circuit_code[i+1] == "(":
                    if i == final_i:
                        self.add_elem_to_elements(
                            elem_str=circuit_code[i],
                            node_1=current_node,
                            node_2=0,
                            elements=elements,
                            main_loop=not bool(len(hold_nodes)),
                        )
                        hold_nodes.append(0)
                    else:
                        idx += 1
                        self.add_elem_to_elements(
                            elem_str=circuit_code[i],
                            node_1=current_node,
                            node_2=idx,
                            elements=elements,
                            main_loop=not bool(len(hold_nodes)),
                        )
                        hold_nodes.append(idx)

                elif circuit_code[i+1] == ")":
                    next_node = hold_nodes.pop()
                    self.add_elem_to_elements(
                        elem_str=circuit_code[i],
                        node_1=current_node,
                        node_2=next_node,
                        elements=elements,
                        main_loop=not bool(len(hold_nodes)),
                    )
                    current_node = next_node

                else:
                    idx += 1
                    self.add_elem_to_elements(
                        elem_str=circuit_code[i],
                        node_1=current_node,
                        node_2=idx,
                        elements=elements,
                        main_loop=not bool(len(hold_nodes)),
                    )
                    current_node = idx

        self.add_all_to_all_cap_to_elements(idx+1, elements)

        return sq.Circuit(elements, flux_dist='junctions')
