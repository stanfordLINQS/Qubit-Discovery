from collections import defaultdict
from typing import Dict, List, Optional, Union

from scipy.stats import uniform, loguniform
import numpy as np
import torch

import SQcircuit as sq

from SQcircuit.circuit import Circuit
from SQcircuit.elements import Capacitor, Element, Inductor, Junction, Loop
from SQcircuit.settings import get_optim_mode


class CircuitSampler:
    """Class used to randomly sample different circuit configurations with a
    single inductive loop.

    Parameters
    ----------
        capacitor_range:
            A list specifying the lower bound and upper bound for the
            capacitors, in Farads.
        inductor_range:
            A list specifying the lower bound and upper bound for the inductors,
            in Henries.
        junction_range:
            A list specifying the lower bound and upper bound for the junctions,
            in Hertz.
        flux_range:
            A range of external flux values to uniformly sample from for the
            loop, in units of Phi_0/2π.
        elems_not_to_optimize:
            A list of element types no to optimize. These are randomly sampled,
            but then fixed during the optimization procedure (the 
            ``required_grad`` attribute is set to False). If ``None``, all
            element types are optimized.
    """

    def __init__(
        self,
        capacitor_range: List[float],
        inductor_range: List[float],
        junction_range: List[float],
        flux_range: List[float],
        elems_not_to_optimize: Optional[List[Union[Element, Loop]]]=None,
    ) -> None:

        self.capacitor_range = capacitor_range
        self.inductor_range = inductor_range
        self.junction_range = junction_range
        self.flux_range = flux_range

        if elems_not_to_optimize is None:
            elems_not_to_optimize = []
        self.elems_to_optimize = [
            elem_type for elem_type in [Capacitor, Inductor, Junction, Loop]
            if elem_type not in elems_not_to_optimize
        ]

        # Special circuit codes used to create symmetric circuits
        self.special_circuit_codes = [
            'transmon',
            'flux_qubit',
            'JJJJ_1',
            'JJJJ_2'
        ]

    @property
    def bounds(self) -> Dict[Union[Element, Loop], Union[torch.Tensor, List[float]]]:
        """
        Bounds for circuit parameters, as a dictionary of
        ``element type: (lower bound, upper_bound)``. To match ``SQcircuit``, the
        units are
            - Capacitors: Farads
            - Inductors: Henries
            - Junctions: 2π*Hz
            - External flux: Phi_0

        The units of junction and external flux differ from those used to
        initialize the sampler. 
        """

        # Bounds in new units to represent difference between setting
        # and getting value in `SQcircuit.`
        flux_range_bounds = [i * 2 * np.pi for i in self.flux_range]
        junction_range_bounds = [i * 2 * np.pi for i in self.junction_range]

        if get_optim_mode():
            return {
                Capacitor: torch.tensor(self.capacitor_range, dtype=torch.float64),
                Inductor: torch.tensor(self.inductor_range, dtype=torch.float64),
                Junction: torch.tensor(junction_range_bounds, dtype=torch.float64),
                Loop: torch.tensor(flux_range_bounds, dtype=torch.float64)
            }
        else:
            return {
                Capacitor: self.capacitor_range,
                Inductor: self.inductor_range,
                Junction: junction_range_bounds,
                Loop: flux_range_bounds
            }

    def _get_elem(
        self,
        elem_str: str,
        main_loop: bool,
    ) -> Union[Capacitor, Inductor, Junction]:
        """Get a random capacitor, inductor, or junction based on the specified
        range for sampler. Element values are log-uniformly sampled.

        Parameters
        ----------
            elem_str:
                A string element type that should be either ``"L"``,``"C"``,
                or ``"J"``.
            main_loop:
                A boolean that indicates whether we are in the main loop of
                the circuit or not.

        Returns
        ----------
            An element of type specified by``elem_str``, which includes the
            loop if ``main_loop`` is ``True``.
        """

        loops = []
        if main_loop:
            loops = [self.loop,]

        if elem_str == 'J':
            junc_value = loguniform.rvs(*self.junction_range)
            elem = sq.Junction(
                junc_value,
                'Hz',
                loops=loops,
                requires_grad=get_optim_mode() and Junction in self.elems_to_optimize,
            )
        elif elem_str == 'L':
            ind_value = loguniform.rvs(*self.inductor_range)
            elem = sq.Inductor(
                ind_value,
                'H',
                loops=loops,
                requires_grad=get_optim_mode() and Inductor in self.elems_to_optimize,
            )
        elif elem_str == 'C':
            cap_value = loguniform.rvs(*self.capacitor_range)
            elem = Capacitor(
                cap_value,
                'F',
                requires_grad=get_optim_mode() and Capacitor in self.elems_to_optimize,
            )
        else:
            raise ValueError("elem_str should be either 'J', 'L', or 'C' ")

        return elem

    def _sample_special_circuit(self, circuit_code):
        """Sample a symmetric circuit.
        
        Parameters
        """
        if circuit_code == 'transmon':
            junc_1 = self._get_elem('J', main_loop=True)
            junc_2 = junc_1

            cap = self._get_elem('C', main_loop=False)

            elements = {(0, 1): [junc_1, junc_2, cap]}

        elif circuit_code == 'flux_qubit':
            junc_1 = self._get_elem('J', main_loop=True)
            junc_2 = junc_1
            junc_3 = self._get_elem('J', main_loop=True)

            cap_1 = self._get_elem('C', main_loop=False)
            cap_2 = cap_1
            cap_3 = self._get_elem('C', main_loop=False)

            elements = {
                (0, 1): [junc_1, cap_1],
                (1, 2): [junc_2, cap_2],
                (2, 0): [junc_3, cap_3]
            }

        elif circuit_code == 'JJJJ_1':

            junc_1 = self._get_elem('J', main_loop=True)
            junc_2 = self._get_elem('J', main_loop=True)
            junc_3 = junc_2
            junc_4 = junc_3

            cap_1 = self._get_elem('C', main_loop=False)
            cap_2 = self._get_elem('C', main_loop=False)
            cap_3 = cap_2
            cap_4 = cap_3

            elements = {
                (0, 1): [junc_1, cap_1],
                (1, 2): [junc_2, cap_2],
                (2, 3): [junc_3, cap_3],
                (3, 0): [junc_4, cap_4]
            }

        elif circuit_code == 'JJJJ_2':

            junc_1 = self._get_elem('J', main_loop=True)
            junc_2 = self._get_elem('J', main_loop=True)
            junc_3 = junc_1
            junc_4 = junc_2

            cap_1 = self._get_elem('C', main_loop=False)
            cap_2 = self._get_elem('C', main_loop=False)
            cap_3 = cap_1
            cap_4 = cap_2

            elements = {
                (0, 1): [junc_1, cap_1],
                (1, 2): [junc_2, cap_2],
                (2, 3): [junc_3, cap_3],
                (3, 0): [junc_4, cap_4]
            }

        else:
            raise ValueError('The circuit code is not supported!')

        return sq.Circuit(elements, flux_dist='junctions')

    def _add_elem_to_elements(
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

        elem = self._get_elem(elem_str, main_loop)

        if (node_2, node_1) in elements:
            elements[(node_2, node_1)].append(elem)
        else:
            elements[(node_1, node_2)].append(elem)

    @staticmethod
    def _if_capacitor_between_nodes(
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

    def _add_all_to_all_cap_to_elements(
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
                if self._if_capacitor_between_nodes(i, j, elements):
                    continue

                self._add_elem_to_elements(
                    elem_str='C',
                    node_1=i,
                    node_2=j,
                    elements=elements,
                )

    @staticmethod
    def _find_last_element_index_outside_parentheses(circuit_code: str) -> int:
        """Find last element index outside parentheses. This is the last
        element that encloses the main loop.

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

    def sample_circuit_code(self, circuit_code: str) -> Circuit:
        """Sample a random circuit with specified circuit code. Element
        values are log-uniformly sampled from the element ranges used to
        initialize the class.

        Parameters
        ----------
            circuit_code:
                A string specifying the circuit code.

        Returns
        ----------
            The randomly sampled circuit.
        """

        # Initialize the loop the loop
        flux_value = uniform.rvs(loc=self.flux_range[0],
                                 scale=self.flux_range[1] - self.flux_range[0])
        self.loop = Loop(
            flux_value,
            requires_grad=get_optim_mode() and Loop in self.elems_to_optimize
        )

        if circuit_code in self.special_circuit_codes:
            return self._sample_special_circuit(circuit_code)

        current_node = 0
        hold_nodes = []
        idx = 0
        elements = defaultdict(list)

        final_i = self._find_last_element_index_outside_parentheses(circuit_code)

        for i in range(len(circuit_code)):
            if circuit_code[i] in ("J", "L", "C"):
                if i == final_i and i == len(circuit_code)-1:
                    self._add_elem_to_elements(
                        elem_str=circuit_code[i],
                        node_1=current_node,
                        node_2=0,
                        elements=elements,
                        main_loop=not bool(len(hold_nodes)),
                    )
                    continue
                if circuit_code[i+1] == "(":
                    if i == final_i:
                        self._add_elem_to_elements(
                            elem_str=circuit_code[i],
                            node_1=current_node,
                            node_2=0,
                            elements=elements,
                            main_loop=not bool(len(hold_nodes)),
                        )
                        hold_nodes.append(0)
                    else:
                        idx += 1
                        self._add_elem_to_elements(
                            elem_str=circuit_code[i],
                            node_1=current_node,
                            node_2=idx,
                            elements=elements,
                            main_loop=not bool(len(hold_nodes)),
                        )
                        hold_nodes.append(idx)
                elif circuit_code[i+1] == ")":
                    next_node = hold_nodes.pop()
                    self._add_elem_to_elements(
                        elem_str=circuit_code[i],
                        node_1=current_node,
                        node_2=next_node,
                        elements=elements,
                        main_loop=not bool(len(hold_nodes)),
                    )
                    current_node = next_node
                else:
                    idx += 1
                    self._add_elem_to_elements(
                        elem_str=circuit_code[i],
                        node_1=current_node,
                        node_2=idx,
                        elements=elements,
                        main_loop=not bool(len(hold_nodes)),
                    )
                    current_node = idx

        self._add_all_to_all_cap_to_elements(idx+1, elements)

        return sq.Circuit(elements, flux_dist='junctions')
