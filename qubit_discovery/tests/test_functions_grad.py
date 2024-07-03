from SQcircuit.tests.test_grad import function_grad_test
from SQcircuit.tests.conftest import (
    create_fluxonium_numpy,
    create_fluxonium_torch_flux,
    create_flux_transmon_numpy,
    create_flux_transmon_torch
)

from qubit_discovery.losses.functions import (
    flux_sensitivity
)


def test_flux_sensitivity_grad() -> None:
    print('Testing fluxonium')
    flux_points = [1e-2, 0.25, 0.5 - 1e-2, 0.5 + 1e-2, 0.75]
    trunc_num = 120

    for phi_ext in flux_points:
        circuit_numpy = create_fluxonium_numpy(trunc_num, phi_ext)
        circuit_torch = create_fluxonium_torch_flux(trunc_num, phi_ext)

        function_grad_test(
            circuit_numpy,
            flux_sensitivity,
            circuit_torch,
            flux_sensitivity,
            num_eigenvalues=50,
            delta=1e-6
        )

    print('Testing transmon')
    trunc_num = 50

    for phi_ext in flux_points:
        circuit_numpy = create_flux_transmon_numpy(trunc_num, phi_ext)
        circuit_torch = create_flux_transmon_torch(trunc_num, phi_ext)

        function_grad_test(
            circuit_numpy,
            flux_sensitivity,
            circuit_torch,
            flux_sensitivity,
            num_eigenvalues=50,
            delta=1e-6
        )
