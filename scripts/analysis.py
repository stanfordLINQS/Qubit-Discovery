import numpy as np

import SQcircuit as sq
import SQcircuit.functions as sqf
from SQcircuit.settings import get_optim_mode


################################################################################
# Initialize circuit
################################################################################

def reset_loop_flux(circuit, flux):
    for loop in circuit.loops:
        loop.set_flux(flux)


def reset_charge_modes(circuit, default_charge=0):
    for charge_island_idx in circuit.charge_islands.keys():
        charge_mode = charge_island_idx + 1
        circuit.set_charge_offset(charge_mode, default_charge)

################################################################################
# Print element values
################################################################################


def round_n_sigfigs(value):
    scale = np.floor(np.log10(value))
    value *= np.power(10, -scale)
    value = np.round(value, 3)
    value /= np.power(10, -scale)
    return value


def get_default_unit(element):
    if type(element) is sq.Junction:
        return 'GHz'
    elif type(element) is sq.Capacitor:
        return 'F'
    elif type(element) is sq.Inductor:
        return 'H'
    return ''


def build_circuit_topology_string(circuit):
    output = "\n"
    for circuit_edge, elements in circuit.elements.items():
        output += f"Edge {circuit_edge}\n"

        for element_idx, element in enumerate(elements):
            output += f"\tElement {element_idx}: {type(element)}\n"
            element_unit = get_default_unit(element)
            element_value = element.get_value(element_unit)
            if type(element) is sq.Junction:
                element_value /= (2 * np.pi)
            if get_optim_mode():
                element_value = element_value.detach().numpy()
            element_value = round_n_sigfigs(element_value)
            output += f"\tValue: {element_value} {element_unit}\n"
        output += "\n"
    return output


################################################################################
# Plot charge spectrum
################################################################################
def sweep_charge_spectrum(
    circuit,
    sweep_charge_node_bools,
    n_eig=4,
    charge_min=0,
    charge_max=1,
    default_charge=0,
    n_charge_points=20
):
    circuit_initial_flux = circuit.loops[0].value() / (2 * np.pi)
    reset_loop_flux(circuit, circuit_initial_flux)

    n_g_vals = np.linspace(charge_min, charge_max, n_charge_points)
    spectrum = np.zeros((n_eig, n_charge_points))

    if sum(circuit.omega == 0) == 0:
        return None
    else:
        print("Calculating charge spectrum...")
        for charge_sweep_idx, n_g in enumerate(n_g_vals):
            print(f"{charge_sweep_idx+1}/{n_charge_points}")
            for charge_island_idx in circuit.charge_islands.keys():
                if sweep_charge_node_bools[charge_island_idx]:
                    circuit.set_charge_offset(
                        charge_island_idx + 1, n_g
                    )
                else:
                    circuit.set_charge_offset(
                        charge_island_idx + 1, default_charge
                    )

            # circuit.update() # necessary?
            x, _ = circuit.diag(n_eig)
            spectrum[:, charge_sweep_idx] = sqf.to_numpy(x)

    # Set zeroth eigenfrequency to zero
    spectrum -= spectrum[0, :]
    return n_g_vals, spectrum


def plot_1d_charge_spectrum(n_g_vals, spectrum, ax, n_eig=4):
    for eigen_idx in range(n_eig):
        ax.plot(n_g_vals, spectrum[eigen_idx, :])

    ax.set_xlabel(r"$n_g$")
    ax.set_ylabel(r"$f_i-f_0$ [GHz]")

################################################################################
# Plot flux spectrum
################################################################################


def calculate_flux_spectrum(
    circuit,
    flux_min=0,
    flux_max=0.5,
    n_eig=4,
    count=30,
):
    reset_charge_modes(circuit)

    # Currently assume one loop
    initial_flux_value = circuit.loops[0].value() / (2 * np.pi)
    flux_values = np.linspace(flux_min, flux_max, count)
    flux_spectrum = np.zeros((n_eig, count))
    print("Calculating flux spectrum...")
    for flux_idx, flux in enumerate(flux_values):
        print(f"{flux_idx+1}/{count}")
        for loop in circuit.loops:
            loop.set_flux(flux)
        circuit.update()
        eigenvalues, _ = circuit.diag(n_eig)
        flux_spectrum[:, flux_idx] = sqf.to_numpy(eigenvalues)
        flux_spectrum -= flux_spectrum[0, :]

    # Reset circuit
    for loop in circuit.loops:
        loop.set_flux(initial_flux_value)
    circuit.update()
    circuit.diag(n_eig)
    return flux_values, flux_spectrum


def plot_flux_spectrum(flux_vals, spectrum, ax, n_eig=-1):
    if n_eig == -1:
        n_eig = np.shape(spectrum)[0]
    for eigen_idx in range(n_eig):
        ax.plot(flux_vals, spectrum[eigen_idx, :])

    ax.set_xlabel(r"$\Phi_{ext}/\Phi_0$")
    ax.set_ylabel(r"$f_i-f_0$ [GHz]")
