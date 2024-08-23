import qubit_discovery as qd
import SQcircuit as sq

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

    circuit.diag(10)

    qd.optimization.assign_trunc_nums(circuit, total_trunc_num=K)

    assert circuit.trunc_nums == [9, 77]

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

    K = 700
    circuit.truncate_circuit(K)
    assert circuit.trunc_nums == [13, 13]

    circuit.diag(10)

    qd.optimization.assign_trunc_nums(circuit, total_trunc_num=K)

    assert circuit.trunc_nums == [13, 13]
