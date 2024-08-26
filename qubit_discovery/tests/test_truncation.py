import SQcircuit as sq

import qubit_discovery as qd
from qubit_discovery.optimization.truncation import (
    assign_trunc_nums,
    test_convergence as convergence_test # rename so pytest ignores
)

N_EIG = 10

def test_JLL():
    loop = sq.Loop(0.35)
    circuit = sq.Circuit(
        {
            (0, 1): [sq.Junction(61, 'GHz', loops=[loop]), sq.Capacitor(4e-13, 'F')],
            (1, 2): [sq.Inductor(5e-9, 'H', loops=[loop]), sq.Capacitor(4e-13, 'F')],
            (2, 0): [sq.Inductor(1e-9, 'H', loops=[loop]), sq.Capacitor(1e-13, 'F')],
        },
        flux_dist='junctions'
    )

    K = 700
    circuit.truncate_circuit(K)
    circuit.diag(N_EIG)

    assert circuit.trunc_nums == [26, 26]
    assert not circuit.check_convergence(t=5)[0], 'Circuit should not converge'

    qd.optimization.assign_trunc_nums(circuit, total_trunc_num=K)
    circuit.diag(N_EIG)

    assert circuit.trunc_nums == [9, 77]
    assert circuit.check_convergence(t=5)[0], 'Circuit should converge'


def test_JJJ_even():
    """Test the default behavior in the case of charge modes (which is to 
    just assign truncation numbers evenly).
    """
    J = sq.Junction(5, 'GHz')
    C = sq.Capacitor(1, 'GHz')
    circuit = sq.Circuit(
        {
            (0, 1): [J, C],
            (1, 2): [J, C],
            (2, 0): [J, C],
        }
    )

    K = 1000
    circuit.truncate_circuit(K)
    assert circuit.trunc_nums == [16, 16]

    circuit.diag(N_EIG)

    qd.optimization.assign_trunc_nums(circuit, total_trunc_num=K)

    assert circuit.trunc_nums == [16, 16]


def build_paper_circuit():
    loop = sq.Loop(0.5)

    JJ_10 = sq.Junction(10.83, 'GHz', loops=[loop])
    C_10 = sq.Capacitor(7.974e-15, 'F')
    L_12 = sq.Inductor(1.53e-06, 'H', loops=[loop])
    C_12 = sq.Capacitor(3.989e-14, 'F')
    JJ_23 = sq.Junction(33.96, 'GHz', loops=[loop])
    C_23 = sq.Capacitor(4.79e-14, 'F')
    L_03 = sq.Inductor(5.482e-13, 'H', loops=[loop])
    C_03 = sq.Capacitor(4.806e-15, 'F')
    C_02 = sq.Capacitor(6.298e-13, 'F')
    C_13 = sq.Capacitor(3.32e-13, 'F')

    elements = {
        (0, 1): [JJ_10, C_10],
        (1, 2): [L_12, C_12],
        (2, 3): [JJ_23, C_23],
        (0, 3): [L_03, C_03],
        (0, 2): [C_02],
        (1, 3): [C_13]
    }

    return sq.Circuit(elements)

def test_paper_circuit() -> None:
    circuit = build_paper_circuit()
    K = 6000
    circuit.truncate_circuit(K)
    print('Even truncation', circuit.m)
    assert circuit.m == [18, 18, 17]

    circuit.diag(N_EIG)
    assign_trunc_nums(
        circuit,
        K,
        min_trunc_harmonic=4
    )
    print('Heuristic truncation', circuit.m)
    assert circuit.m == [4, 88, 17]

    # Need to set back the truncation numbers to match what we had
    # during diagonalization.
    circuit.truncate_circuit(K)

    assign_trunc_nums(
        circuit,
        K,
        min_trunc_harmonic=4,
        min_trunc_charge=12
    )
    print('Heuristic truncation with charge mininum', circuit.m)
    assert circuit.m == [4, 65, 23]

    circuit.diag(N_EIG)
    passed_test, test_values = convergence_test(circuit, eig_vec_idx=1)
    print('Convergence test values', test_values)
    assert passed_test, "Test should pass"
