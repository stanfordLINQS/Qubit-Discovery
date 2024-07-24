"""test_sampler contains tests for generating random circuit topologies with
circuit values sampled from a fixed range."""

import SQcircuit as sq
import numpy as np

from SQcircuit import Circuit

from qubit_discovery.optimization.sampler import CircuitSampler


def test_convergence():
    # Create fluxonium circuit
    def create_fluxonium(
        cap_value, cap_unit, ind_value, ind_unit,
        junction_value, junction_unit
    ):
        loop = sq.Loop()
        loop.set_flux(0)

        capacitor = sq.Capacitor(cap_value, cap_unit, Q=1e6)
        inductor = sq.Inductor(ind_value, ind_unit, Q=500e6, loops=[loop])
        junction = sq.Junction(
            junction_value, junction_unit, cap=capacitor,
            A=1e-7, x=3e-06, loops=[loop]
        )
        circuit_fluxonium = sq.Circuit(
            {(0, 1): [capacitor, inductor, junction], }
        )
        return circuit_fluxonium
    trunc_cutoff = 100
    trunc_range = np.arange(12, trunc_cutoff + 1)
    cutoff = 3

    num_eigenvalues = 10
    for x in trunc_range:
        fluxonium = create_fluxonium(
            2, 'GHz', 0.46,
            'GHz', 10.2, 'GHz'
        )
        # Assuming contiguous divergence/convergence trunc nums, find the lowest
        # truncation number that converges
        fluxonium.set_trunc_nums([x, ])
        fluxonium.diag(num_eigenvalues)
        if not fluxonium.check_convergence(threshold=1e-5)[0]:
            cutoff = x + 1
    assert cutoff == 41


def get_elements_from_code(circuit_code: str) -> Circuit:

    sampler = CircuitSampler(
        capacitor_range=[12e-15, 12e-9],
        inductor_range=[12e-9, 12e-6],
        junction_range=[1e9, 10e9],
        flux_range=[0.5, 0.5],
        elems_not_to_optimize=[sq.Loop]
    )
    circuit = sampler.sample_circuit_code(circuit_code)

    return circuit.elements


def test_circuit_sampling_from_code_1() -> None:

    elements = get_elements_from_code("JL(JC)")

    elem1, elem2, elem3 = elements[(0, 1)]
    assert isinstance(elem1, sq.Junction)
    assert isinstance(elem2, sq.Inductor)
    assert isinstance(elem3, sq.Capacitor)
    assert elem1.loops != []
    assert elem2.loops != []

    elem1, elem2 = elements[(1, 2)]
    assert isinstance(elem1, sq.Junction)
    assert isinstance(elem2, sq.Capacitor)
    assert elem1.loops == []

    elem, = elements[(2, 0)]
    assert isinstance(elem, sq.Capacitor)


def test_circuit_sampling_from_code_2() -> None:

    elements = get_elements_from_code("J(JC)J(LC)")

    elem1, elem2, elem3 = elements[(0, 1)]
    assert isinstance(elem1, sq.Junction)
    assert isinstance(elem2, sq.Junction)
    assert isinstance(elem3, sq.Capacitor)
    assert elem1.loops != []
    assert elem2.loops != []

    elem1, elem2 = elements[(0, 2)]
    assert isinstance(elem1, sq.Junction)
    assert isinstance(elem2, sq.Capacitor)
    assert elem1.loops == []

    elem, = elements[(2, 1)]
    assert isinstance(elem, sq.Capacitor)

    elem1, elem2 = elements[(1, 3)]
    assert isinstance(elem1, sq.Inductor)
    assert isinstance(elem2, sq.Capacitor)
    assert elem1.loops == []

    elem, = elements[(3, 0)]
    assert isinstance(elem, sq.Capacitor)

    elem, = elements[(2, 3)]
    assert isinstance(elem, sq.Capacitor)


def test_circuit_sampling_from_code_3() -> None:

    elements = get_elements_from_code("JJ(J(LC)C)")

    elem1, elem2, elem3 = elements[(0, 1)]
    assert isinstance(elem1, sq.Junction)
    assert isinstance(elem2, sq.Junction)
    assert isinstance(elem3, sq.Capacitor)
    assert elem1.loops != []
    assert elem2.loops != []

    elem1, elem2 = elements[(1, 2)]
    assert isinstance(elem1, sq.Junction)
    assert isinstance(elem2, sq.Capacitor)
    assert elem1.loops == []

    elem1, elem2 = elements[(1, 3)]
    assert isinstance(elem1, sq.Inductor)
    assert isinstance(elem2, sq.Capacitor)

    elem, = elements[(3, 2)]
    assert isinstance(elem, sq.Capacitor)

    elem, = elements[(2, 0)]
    assert isinstance(elem, sq.Capacitor)

    elem, = elements[(0, 3)]
    assert isinstance(elem, sq.Capacitor)


def test_circuit_sampling_from_code_4() -> None:

    sq.set_optim_mode(True)

    elements = get_elements_from_code("flux_qubit")

    junc_1, cap_1 = elements[(0, 1)]
    junc_2, cap_2 = elements[(1, 2)]
    junc_3, cap_3 = elements[(2, 0)]

    assert junc_1 == junc_2
    assert cap_1 == cap_2
    assert junc_2 != junc_3
    assert cap_2 != cap_3

    circuit = sq.Circuit(elements, flux_dist="junctions")

    assert len(circuit.parameters) == 4

    sq.set_optim_mode(False)


def test_circuit_sampling_from_code_5() -> None:

    sq.set_optim_mode(True)

    elements = get_elements_from_code("transmon")

    junc_1, junc_2, cap = elements[(0, 1)]

    assert junc_1 == junc_2

    circuit = sq.Circuit(elements, flux_dist="junctions")

    circuit.description()

    assert len(circuit.parameters) == 2

    sq.set_optim_mode(False)


def test_circuit_sampling_from_code_6() -> None:

    sq.set_optim_mode(True)

    elements = get_elements_from_code("JJJJ_1")

    junc_2, cap_2 = elements[(1, 2)]
    junc_3, cap_3 = elements[(2, 3)]

    assert junc_2 == junc_3
    assert cap_2 == cap_3

    circuit = sq.Circuit(elements, flux_dist="junctions")

    circuit.description()

    assert len(circuit.parameters) == 4

    sq.set_optim_mode(False)


def test_circuit_sampling_from_code_7() -> None:

    sq.set_optim_mode(True)

    elements = get_elements_from_code("JJJJ_2")

    junc_1, cap_1 = elements[(0, 1)]
    junc_3, cap_3 = elements[(2, 3)]

    assert junc_1 == junc_3
    assert cap_1 == cap_3

    circuit = sq.Circuit(elements, flux_dist="junctions")

    circuit.description()

    assert len(circuit.parameters) == 4

    sq.set_optim_mode(False)
