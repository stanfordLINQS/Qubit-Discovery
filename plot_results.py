# Optionally plot circuit spectrum if desired
# spectrum_optimal_circuit(loss_record, codename="Fluxonium")

from functions import lookup_codename, get_element_counts

import argparse
from matplotlib import pyplot as plt
import joblib

# Assign keyword arguments
parser = argparse.ArgumentParser()
parser.add_argument("codenames")
parser.add_argument("id")
args = parser.parse_args()

# TODO: Consolidate following definition with that in loss.py
omega_target = 1 # GHz

codenames = ["JL", ]
num_runs = 1

def load_loss_record(url):
    # for codename in codenames:
    #     for run_idx in range(num_runs):
    loss_record = joblib.load(url)
    return loss_record

def get_optimal_key(loss_record, codename=None):
  optimal_loss = 1e100
  optimal_key = None

  for circuit, codename_key, _ in loss_record.keys():
    key = (circuit, codename_key, 'Total Loss')
    loss = loss_record[key][-1]
    if loss < optimal_loss and (codename in key or codename is None):
        optimal_loss = loss
        optimal_key = key

  return optimal_key

def plot_results(loss_record):
    plot_scheme = {'Transmon': 'b', 'Fluxonium': 'darkorange',
                   'JJJ': 'tab:purple', 'JJL': 'c', 'JLL': 'g'}

    fig, axs = plt.subplots(2, 3, figsize=(22, 11))

    # fig.suptitle('Scaling from N=2 to N=3 Inductive Elements')

    def plot_circuit_metrics(circuit, loss_record, codename, optimal_keys,
                             show_label=False):
        label = codename if show_label else None
        axs[0, 0].set_title(f'Total Loss')
        axs[1, 0].set_title(f'Frequency')
        axs[0, 1].set_title(f'Flux Sensitivity')
        axs[1, 1].set_title(f'Charge Sensitivity')
        axs[0, 2].set_title(f'Anharmonicity')
        axs[1, 2].set_title(r'$T_1$')

        axs[0, 0].set_yscale('log')
        axs[0, 1].set_yscale('log')
        axs[0, 2].set_yscale('log')
        axs[1, 0].set_yscale('log')
        axs[1, 1].set_yscale('log')
        axs[1, 2].set_yscale('log')
        key = (circuit, codename, 'Total Loss')
        if key in optimal_keys:
            alpha = None
            linestyle = None
        else:
            alpha = 0.3
            linestyle = '--'

        axs[0, 0].plot(loss_record[(circuit, codename, 'Total Loss')],
                       plot_scheme[codename], label=label, alpha=alpha,
                       linestyle=linestyle)
        axs[0, 0].legend(loc='upper right')
        axs[1, 0].plot(loss_record[(circuit, codename, 'omega')],
                       plot_scheme[codename], label=label, alpha=alpha,
                       linestyle=linestyle)
        axs[1, 0].legend(loc='lower right')
        axs[0, 1].plot(loss_record[(circuit, codename, 'flux_sensitivity')],
                       plot_scheme[codename], label=label, alpha=alpha,
                       linestyle=linestyle)
        axs[0, 1].legend(loc='upper right')
        axs[1, 1].plot(loss_record[(circuit, codename, 'charge_sensitivity')],
                       plot_scheme[codename], label=label, alpha=alpha,
                       linestyle=linestyle)
        axs[1, 1].legend(loc='center right')
        axs[0, 2].plot(loss_record[(circuit, codename, 'A')],
                       plot_scheme[codename], label=label, alpha=alpha,
                       linestyle=linestyle)
        axs[0, 2].legend(loc='upper right')
        axs[1, 2].plot(loss_record[(circuit, codename, 'T1')],
                       plot_scheme[codename], label=label, alpha=alpha,
                       linestyle=linestyle)
        axs[1, 2].legend(loc='lower left')

    axs[1, 0].axhline(y=omega_target, color='m', linestyle=':')
    axs[0, 2].axhline(y=22, color='m', linestyle=':')

    optimal_keys = [get_optimal_key(loss_record, codename=codename) for codename
                    in codenames]

    plotted_codenames = set()
    for circuit in loss_record.keys():
        show_label = False
        junction_count, inductor_count, _ = get_element_counts(circuit)
        codename = lookup_codename(junction_count, inductor_count)
        if codename not in plotted_codenames:
            show_label = True
            plotted_codenames.add(codename)
        plot_circuit_metrics(circuit, loss_record, codename, optimal_keys,
                             show_label=show_label)

    plt.show()

def main():
    circuit_codes = args.codenames
    count = int(args.count)
    for
    loss_record = load_loss_record('/home/mckeehan/sqcircuit/Qubit-Discovery/results/loss_record.pickle')
    plot_results(loss_record)


if __name__ == "__main__":
    main()