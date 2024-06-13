import matplotlib.pyplot as plt
import numpy as np

import SQcircuit as sq
import SQcircuit.functions as sqf


#############################
# Initialize circuit        #
#############################

def reset_loop_flux(circuit, flux):
    for loop in circuit.loops:
        loop.set_flux(flux)


def reset_charge_modes(circuit, default_charge = 0):
    for charge_island_idx in circuit.charge_islands.keys():
        charge_mode = charge_island_idx + 1
        circuit.set_charge_offset(charge_mode, default_charge)

#############################
# Print element values      #
#############################


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
            output += f"Element {element_idx}: {type(element)}\n"
            element_unit = get_default_unit(element)
            element_value = element.get_value(element_unit)
            if type(element) is sq.Junction:
                element_value /= (2 * np.pi)
            if sq.get_optim_mode():
                element_value = element_value.detach().numpy()
            element_value = round_n_sigfigs(element_value)
            output += f"Value: {element_value} {element_unit}\n"
        output += "\n"
    return output


#############################
# Plot charge spectrum      #
#############################
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
        for charge_sweep_idx, n_g in enumerate(n_g_vals):
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
            spectrum[:, charge_sweep_idx] = sqf.numpy(x)

    spectrum -= spectrum[0, :] # Set zeroth eigenenergy to zero
    return n_g_vals, spectrum


def grid_charge_spectrum(
    circuit,
    sweep_charge_node_bools,
    n_eig=4,
    charge_min=0,
    charge_max=1,
    default_charge=0,
    n_charge_points=20
):
    if not np.count_nonzero(sweep_charge_node_bools):
        return None

    circuit_initial_flux = circuit.loops[0].value() / (2 * np.pi)

    phis = [np.linspace(charge_min, charge_max, n_charge_points)] * np.count_nonzero(sweep_charge_node_bools)
    phi_meshes = np.meshgrid(*phis) # this gets quite large if many modes

    out = np.zeros_like(phi_meshes[0])

    it = np.nditer(phi_meshes[0], flags=['multi_index'])
    reset_loop_flux(circuit, circuit_initial_flux)

    sweep_modes = [i for i, val in enumerate(sweep_charge_node_bools) if val]

    for _ in it:
        reset_charge_modes(circuit, default_charge)
        for idx, mode_num in enumerate(sweep_modes):
            circuit.set_charge_offset(mode_num + 1, phi_meshes[idx][it.multi_index])
        efreqs , _ = circuit.diag(n_eig)
        efreqs = sqf.numpy(efreqs)
        out[it.multi_index] = efreqs[1] - efreqs[0]
    
    return phi_meshes, out


def plot_1D_charge_spectrum(n_g_vals, spectrum, ax, n_eig=4):
    for eigen_idx in range(n_eig):
        ax.plot(n_g_vals, spectrum[eigen_idx, :])

    ax.set_xlabel(r"$n_g$")
    ax.set_ylabel(r"$f_i-f_0$ [GHz]")


def plot_2D_charge_spectrum(ng1, ng2, frequency, ax):
    c = ax.pcolormesh(ng1, ng2, frequency)
    plt.colorbar(c)


#############################
# Plot flux spectrum        #
#############################
def calculate_flux_spectrum(
    circuit,
    flux_min=0,
    flux_max=1,
    n_eig=4,
    count=200,
    center_count=10,
    delta=0.03
):
    reset_charge_modes(circuit)

    # Currently assume one loop
    initial_flux_value = circuit.loops[0].value() / (2 * np.pi)
    flux_values = np.linspace(flux_min, flux_max, count)
    center_values = np.linspace(0.5 - delta, 0.5 + delta, center_count)
    flux_values = np.sort(np.concatenate((flux_values, center_values)))
    flux_spectrum = np.zeros((n_eig, count + center_count))
    for flux_idx, flux in enumerate(flux_values):
        for loop in circuit.loops:
            loop.set_flux(flux)
        circuit.update()
        eigenvalues, _ = circuit.diag(n_eig)
        flux_spectrum[:, flux_idx] = sqf.numpy(eigenvalues)
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


#############################
# Plot eigenfunctions       #
#############################


def calc_state(
    circuit,
    n,
    num_modes,
    num_points=400
):
    phi_range = []
    for i in range(num_modes):
        phi_range.append(np.linspace(-1, 1, num_points))

    return phi_range, circuit.eig_phase_coord(n, grid=phi_range)


def plot_state_phase(phi_range, 
                     state, 
                     ax,
                     plot_type='abs'):
    ax.set_xlabel(r'$\phi_1$')
    ax.set_ylabel(r'$\phi_2$')

    if plot_type == 'abs':
        ax.pcolor(phi_range[0], phi_range[1], np.abs(state)**2,
                   cmap="binary", shading='auto')
        ax.set_title(r'State Magnitude $|\psi(\phi_1,\phi_2)|^2$')
    if plot_type == 'real':
        ax.pcolor(phi_range[0], phi_range[1], np.real(state),
                   cmap="binary", shading='auto')
        ax.set_title(r'Re$[\psi(\phi_1,\phi_2)]$')
    if plot_type == 'imag':
        ax.pcolor(phi_range[0], phi_range[1], np.imag(state),
                   cmap="binary", shading='auto')
        ax.set_title(r'Im$[\psi(\phi_1,\phi_2)]$')