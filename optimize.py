"""Contains code for optimization of single circuit instance."""

from functions import (
    print_new_circuit_sampled_message
)

from loss import calculate_total_loss

import numpy as np

def assign_settings():
    seed = 2
    random.seed(seed)
    np.random.seed(seed)
    k = 2  # number of circuits to sample of each topology
    num_epochs = 15  # number of training iterations
    lr = 1e-1  # learning rate
    num_eigenvalues = 3
    total_trunc_num = 1000

    omega_target = 0.64  # GHz

    # Target parameter range
    capacitor_range = [1e-15, 12e-12]  # F
    inductor_range = [2e-8, 5e-6]  # H
    junction_range = [1e9 * 2 * np.pi, 100e9 * 2 * np.pi]  # Hz

    gradient_clipping = True
    loss_normalization = False
    learning_rate_scheduler = True
    scheduler_decay_rate = 0.98
    gradient_clipping_threshold = 0.5
    gc_norm_type = 'inf'

    anharmonicity_loss_function = anharmonicity_loss
    log_loss = False
    nesterov_momentum = True
    momentum_value = 0.9

    element_verbose = False
    show_charge_spectrum_plot = False
    show_flux_spectrum_plot = False

    circuit_patterns = ['JJ', 'JL', 'JJJ', 'JJL', 'JLL']
    circuit_patterns.reverse()


def optimize(circuit_code):
    assign_settings()
    sampler = create_sampler(2, capacitor_range, inductor_range, junction_range)
    circuit = sampler.sample_circuit_code(circuit_code)
    # print_new_circuit_sampled_message()
    trunc_nums = circuit.truncate_circuit(total_trunc_num)

    # Get junction/inductor counts in circuit to ascribe codename
    # Note we need a better system for higher element numbers, as for N>3
    # there are topologies with equal numbers of inductors and junctions
    # but different topologies.
    junction_count, inductor_count, _ = get_element_counts(circuit)
    codename = lookup_codename(junction_count, inductor_count)
    # print(f"Circuit codename: {codename}")

    circuit_metadata = get_circuit_metadata(circuit, global_circuit_count)
    all_circuit_metadata += [circuit_metadata, ]
    loss_record = init_loss_record(circuit_metadata, codename)

    circuit.diag(num_eigenvalues)
    if loss_normalization:
        scale_omega = frequency_loss(circuit,
                                     omega_target=omega_target).detach()
        scale_A = anharmonicity_loss_function(circuit).detach()
        scale_T1 = calculate_T1_decoherence(circuit).detach()
        scale_flux_sensitivity = flux_sensitivity_loss(circuit)[0].detach()
        scale_charge_sensitivity = charge_sensitivity_loss(circuit)[
            0].detach()

    converged = True
    # Circuit optimization loop
    for iteration in range(num_epochs):
        optimizer = torch.optim.SGD(
            circuit.parameters,
            nesterov=nesterov_momentum,
            momentum=momentum_value if nesterov_momentum else 0.0,
            lr=lr,
        )
        if show_flux_spectrum_plot:
            plot_circuit_flux_spectrum(circuit, n_eig=num_eigenvalues,
                                       n_phi_ext=20)
        if show_charge_spectrum_plot:
            plot_circuit_charge_spectrum(circuit, n_eig=3,
                                         n_charge_points=20)

        circuit.diag(num_eigenvalues)
        if len(circuit.m) == 1:
            converged = circuit.test_convergence(trunc_nums)
        elif len(circuit.m) == 2:
            trunc_nums = trunc_num_heuristic(circuit,
                                             K=4000,
                                             eig_vec_idx=1,
                                             axes=None)
            circuit.set_trunc_nums(trunc_nums)
            circuit.diag(num_eigenvalues)

            # converged = circuit.test_convergence(trunc_nums)
            converged, _, _ = test_convergence(circuit, eig_vec_idx=1)
        print(f"Converged (true/false)?: {converged}")

        print(f"Omega: {np.abs(circuit.omega)}")
        if not converged:
            print("Warning: Circuit did not converge")
            circuit_batch_idx -= 1
            plt.show()
            break
            # TODO: In addition to breaking, also arXiv circuit

        # Calculate loss, backprop
        loss_freq = frequency_loss(circuit, omega_target=omega_target)
        frequency = first_resonant_frequency(circuit)
        anharmonicity = calculate_anharmonicity(circuit)
        loss_T1 = calculate_T1_decoherence(circuit)
        loss_flux_sensitivity, flux_sensitivity = flux_sensitivity_loss(
            circuit)
        loss_charge_sensitivity, charge_sensitivity = charge_sensitivity_loss(
            circuit)
        loss_anharmonicity = anharmonicity_loss_function(circuit)
        total_loss = calculate_total_loss(loss_freq, loss_anharmonicity,
                                          loss_T1,
                                          loss_flux_sensitivity,
                                          loss_charge_sensitivity,
                                          log_loss=log_loss)

        metrics = (frequency, anharmonicity, 1 / loss_T1, flux_sensitivity,
                   charge_sensitivity, total_loss)
        update_metrics(circuit_metadata, loss_record, metrics)
        print(
            10 * "-" + "\n"
                       f"Iteration: {iteration}",
            f"circ_type: {codename}",
            f"circ_index: {circuit_batch_idx}",
            f"total loss: {total_loss.detach().numpy()}",
            "\n" + 10 * "-" + "\n"
                              f"freq: {frequency}",
            f"freq_loss: {loss_freq.detach().numpy()}",
            f"anharmon: {anharmonicity.detach().numpy()}",
            f"anharmon_loss: {loss_anharmonicity.detach().numpy()}",
            "\n" + 10 * "-" + "\n"
                              f"flux_sens: {flux_sensitivity.detach().numpy()}",
            f"flux_sens_loss: {loss_flux_sensitivity.detach().numpy()}",
            f"charge_sens: {charge_sensitivity.detach().numpy()}",
            f"charge_sens_loss: {loss_charge_sensitivity.detach().numpy()}",
            "\n" + 10 * "-",
            f"T1_loss: {loss_T1.detach().numpy()}",
            "\n" + 10 * "-"
        )

        if loss_normalization:
            loss_freq /= scale_omega
            loss_anharmonicity /= scale_A
            loss_T1 /= scale_T1
            loss_flux_sensitivity /= scale_flux_sensitivity
            loss_charge_sensitivity /= scale_charge_sensitivity
            total_loss = calculate_total_loss(loss_freq, loss_anharmonicity,
                                              loss_T1,
                                              loss_flux_sensitivity,
                                              loss_charge_sensitivity,
                                              log_loss=log_loss)

        total_loss.backward()

        if element_verbose:
            print_circuit_details(circuit)
        for element in list(circuit._parameters.keys()):
            if element_verbose:
                print(60 * "-")
                print(f"Element type: {type(element)}")
                print(f"Element value: {element._value}")
                print(f"Unnormalized element grad: {element._value.grad}")
            element._value.grad *= element._value
            if gradient_clipping:
                # clamp_gradient(element, gradient_clamping_threshold)
                torch.nn.utils.clip_grad_norm_(element._value,
                                               max_norm=gradient_clipping_threshold,
                                               norm_type=gc_norm_type)
            element._value.grad *= element._value
            if learning_rate_scheduler:
                element._value.grad *= (scheduler_decay_rate ** iteration)
            if element_verbose:
                print(f"Normalized element grad: {element._value.grad}")
                print(
                    f"Grad to element ratio: {element._value.grad / element._value}")
        optimizer.step()
        optimizer.zero_grad()
        circuit.update()
        print(210 * "=")

    if not converged:
        # TODO: Save circuit, throw error
        pass

    global_circuit_count += 1

'''loss_record = {}
# all_circuits = []
all_circuit_metadata = []
# for N in [2, ]:

sampler = create_sampler(2, capacitor_range, inductor_range, junction_range)
num_topologies = len(sampler.topologies)
circuit_batch_idx = -1
global_circuit_count = 0
while circuit_batch_idx < k - 1:
    circuit_batch_idx += 1
    # Following block will sample over all circuit topologies for a given N:
    ''''''sampled_circuits = circuit_sampler.sample_one_loop_circuits(
        n=num_topologies,
        with_replacement=False
    )''''''
    # Following block will sample over a single given circuit pattern:
    sampled_circuits = []
    # for circuit_code in circuit_patterns:
    #   sampled_circuits.append(sampler.sample_circuit_code(circuit_code))

    # for circuit_idx, circuit in enumerate(sampled_circuits):
    for circuit_code in circuit_patterns:
        
    # all_circuits += sampled_circuits'''