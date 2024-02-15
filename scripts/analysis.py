from collections import defaultdict
from copy import copy
import os

import dill as pickle
import matplotlib.pyplot as plt
import numpy as np
import SQcircuit as sq
import SQcircuit.functions as sqf
import torch

#############################
# Load circuit              #
#############################

def load_record(url):
    if not os.path.exists(url):
        return None
    file = open(url, 'rb')
    try:
        record = pickle.load(file)
        file.close()
        return record
    except:
        print(f"Failed to read file {url}")


#############################
# Build circuit             #
#############################
        
def element_code_to_class(code):
    if code == 'J':
       return sq.Junction
    if code == 'L':
        return sq.Inductor
    if code == 'C':
        return sq.Capacitor
    return None

def build_circuit(element_dictionary,
                  default_flux=0.5):
    '''
    Element dictionary should be of the form {(0,1): ['J', 3.0, 'GHz], ...}
    '''
    loop = sq.Loop()
    loop.set_flux(default_flux)
    elements = defaultdict(list)
    for edge, edge_element_details in element_dictionary.items():
        for (circuit_type, value, unit) in edge_element_details:
            if circuit_type in ['J', 'L']:
                element = element_code_to_class(circuit_type)(value, unit, loops=[loop, ],
                                                              min_value=0, max_value=1e20)
            else: # 'C'
                element = element_code_to_class(circuit_type)(value, unit,
                                                min_value=0, max_value=1e20)
        elements[edge].append(element)
    
    circuit = sq.Circuit(elements)
    return circuit

def copy_circuit(circuit):
    cap_unit, ind_unit, junction_unit = 'F', 'H', 'Hz'
    loop = sq.Loop()
    loop.set_flux(0.5)

    topology = {}
    all_elements = []
    for edge, elements in circuit.elements.items():
        new_elements = []
        for element in elements:
            element_val = element._value.detach().numpy()
            if type(element) is sq.elements.Capacitor:
                new_element = sq.Capacitor(element_val, cap_unit, Q=1e6, requires_grad=optim_mode)
            elif type(element) is sq.elements.Inductor:
                new_element = sq.Inductor(element_val, ind_unit, Q=500e6, loops=[loop], max_value = 1e-4, requires_grad=optim_mode)
            elif type(element) is sq.elements.Junction:
                new_element = sq.Junction(element_val / (2 * np.pi), junction_unit, max_value = 2 * np.pi * 100e9, loops=[loop], requires_grad=optim_mode)
            new_elements.append(new_element)
            all_elements.append(new_element)
        topology[edge] = new_elements
    print(f"topology: {topology}")

    copy_circuit = sq.Circuit(
        topology,
        flux_dist='all'
        )
    return copy_circuit, all_elements, loop

#############################
# Initialize circuit        #
#############################

def initialize_circuit(circuit: sq.Circuit,
                       num_eigenvalues=10,
                       max_trunc_product=800,
                       default_flux=0.5,
                       default_charge=0):
    print(f"Circuit truncation number: {circuit.ms}")
    ratio = np.power(np.minimum(max_trunc_product / np.prod(circuit.ms), 1), 1 / len(circuit.ms))
    reduced_trunc_nums = np.floor(np.array(circuit.ms) * ratio).astype(np.int64).tolist()
    print(f"Reduced circuit truncation number: {reduced_trunc_nums}")

    circuit.set_trunc_nums(reduced_trunc_nums)
    circuit.set_trunc_nums([10, 10])
    circuit.diag(num_eigenvalues)

    print(f"Previous loop flux value (in units of 2pi): {circuit.loops[0].lpValue / (2 * np.pi)}")

    print(f'Now resetting charge and flux to n_g={default_charge} and phi={default_flux}')
    reset_charge_modes(circuit, default_charge)
    reset_loop_flux(circuit, default_flux)

def reset_loop_flux(circuit,
                    default_flux=0.5):
    for loop in circuit.loops:
        loop.set_flux(default_flux)

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

def print_circuit_elements(circuit):
    for circuit_edge, elements in circuit.elements.items():
        print(f"Edge {circuit_edge}")

    for element_idx, element in enumerate(elements):
        print(f"Element {element_idx}: {type(element)}")
        element_unit = get_default_unit(element)
        element_value = element.get_value(element_unit)
        if type(element) is sq.Junction:
            element_value /= (2 * np.pi)
        if sq.get_optim_mode():
            element_value = element_value.detach().numpy()
        element_value = round_n_sigfigs(element_value)
        print(f"Value: {element_value} {element_unit}")
    print()


#############################
# Plot charge spectrum      #
#############################
def calculate_charge_spectrum(circuit,
                              sweep_charge_node_bools,
                              n_eig=4,
                              charge_min=0,
                              charge_max=1,
                              default_charge=0,
                              n_charge_points=20):
    reset_loop_flux(circuit)

    n_g_vals = np.linspace(charge_min, charge_max, n_charge_points)
    spectrum = np.zeros((n_eig, n_charge_points))

    if sum(circuit.omega == 0) == 0:
        spectrum = circuit.diag(n_eig)
        return None
    else:
        for charge_sweep_idx, n_g in enumerate(n_g_vals):
            for charge_island_idx in circuit.charge_islands.keys():
                charge_mode = charge_island_idx + 1 # SQcircuit uses 1-indexing for charge modes
                if sweep_charge_node_bools[charge_island_idx]:
                    circuit.set_charge_offset(charge_mode, n_g)
                else:
                    circuit.set_charge_offset(charge_mode, default_charge)

            circuit.update() # necessary?
            x, _ = circuit.diag(n_eig)
            spectrum[:, charge_sweep_idx] = sqf.numpy(x)

    spectrum -= spectrum[0, :] # Set zeroth eigenenergy to zero
    return n_g_vals, spectrum

def plot_charge_spectrum(n_g_vals, spectrum, ax, n_eig=4):
    for eigen_idx in range(n_eig):
        ax.plot(n_g_vals, spectrum[eigen_idx, :], 'o')

    ax.set_xlabel(r"Gate Charge $n_g$", fontsize=13)
    ax.set_ylabel(r"$\omega_{i0}$ in GHz", fontsize=13)

#############################
# Plot flux spectrum        #
#############################
def calculate_flux_spectrum(circuit, 
                            flux_min=0,
                            flux_max=1,
                            n_eig=4, count=20):
    reset_charge_modes(circuit)

    flux_values = np.linspace(flux_min, flux_max, count)
    flux_spectrum = np.zeros((n_eig, count))
    for flux_idx, flux in enumerate(flux_values):
        for loop in circuit.loops:
            loop.set_flux(flux)
        circuit.update()
        eigenvalues, _ = circuit.diag(4)
        flux_spectrum[:, flux_idx] = sqf.numpy(eigenvalues)
        flux_spectrum -= flux_spectrum[0, :]
    return flux_values, flux_spectrum

def plot_flux_spectrum(flux_vals, spectrum, ax, n_eig=-1):
    if n_eig == -1:
        n_eig = np.shape(spectrum)[0]
    for eigen_idx in range(n_eig):
        ax.plot(flux_vals, spectrum[eigen_idx, :], 'o')

    ax.set_xlabel(r"Flux Values $\varphi$", fontsize=13)
    ax.set_ylabel(r"$\omega_{i0}$ in GHz", fontsize=13)

#############################
# Plot eigenfunctions       #
#############################

def calc_state(circuit,
               n,
               num_modes,
               num_points=400):
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