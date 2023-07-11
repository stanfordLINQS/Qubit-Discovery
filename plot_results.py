# Optionally plot circuit spectrum if desired
# spectrum_optimal_circuit(loss_record, codename="Fluxonium")

from functions import get_element_counts, code_to_codename

import argparse
from matplotlib import pyplot as plt
import dill as pickle

# Assign keyword arguments
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--codes', type=str, required=True)
parser.add_argument("num_runs")
args = parser.parse_args()

# TODO: Consolidate following definition with that in loss.py
omega_target = 1 # GHz

def load_record(url):
    file = open(url, 'rb')
    record = pickle.load(file)
    file.close()
    return record

def get_optimal_key(loss_record, code=None):
  optimal_loss = 1e100
  optimal_key = None

  for circuit, circuit_code, l in loss_record.keys():
    key = (circuit, circuit_code, 'total_loss')
    loss = loss_record[key][-1]
    if loss < optimal_loss and (code in key or code is None):
        optimal_loss = loss
        optimal_key = key

  return optimal_key

def plot_results(loss_record, circuit_codes, type='metrics'):
    plot_scheme = {'Transmon': 'b', 'Fluxonium': 'darkorange',
                   'JJJ': 'tab:purple', 'JJL': 'c', 'JLL': 'g'}

    fig, axs = plt.subplots(2, 3, figsize=(22, 11))
    metric_titles = ['Total Loss', 'Frequency', 'Flux Sensitivity',
                     'Charge Sensitivity', 'Anharmonicity', r'$T_1$']
    metric_keys = ['total_loss', 'omega', 'flux_sensitivity',
                   'charge_sensitivity', 'A', 'T1']
    loss_titles = ['Frequency Loss', 'Anharmonicity Loss', 'T1 Loss',
                     'Flux Sensitivity Loss', 'Charge Sensitivity Loss', 'Total Loss']
    loss_keys = ['frequency_loss', 'anharmonicity_loss', 'T1_loss',
                   'flux_sensitivity_loss', 'charge_sensitivity_loss', 'total_loss']
    plot_titles = metric_titles if type == 'metrics' else loss_titles
    record_keys = metric_keys if type == 'metrics' else loss_keys

    # fig.suptitle('Scaling from N=2 to N=3 Inductive Elements')

    def plot_circuit_metrics(circuit, loss_record, code, optimal_keys,
                             show_label=False):
        codename = code_to_codename(code)
        label = codename if show_label else None
        for plot_idx in range(6):
            axs[plot_idx % 2, plot_idx // 2].set_title(plot_titles[plot_idx])
            axs[plot_idx % 2, plot_idx // 2].set_yscale('log')

        key = (circuit, code, 'total_loss')
        if key in optimal_keys:
            alpha = None
            linestyle = None
        else:
            alpha = 0.3
            linestyle = '--'

        for plot_idx in range(6):
            axs[plot_idx % 2, plot_idx // 2].plot(loss_record[
                           (circuit, code, record_keys[plot_idx])],
                           plot_scheme[codename], label=label, alpha=alpha,
                           linestyle=linestyle)
        axs[0, 0].legend(loc='upper right')
        axs[1, 0].legend(loc='lower right')
        axs[0, 1].legend(loc='upper right')
        axs[1, 1].legend(loc='center right')
        axs[0, 2].legend(loc='upper right')
        axs[1, 2].legend(loc='lower left')

    axs[1, 0].axhline(y=omega_target, color='m', linestyle=':')
    axs[0, 2].axhline(y=22, color='m', linestyle=':')

    optimal_keys = [get_optimal_key(loss_record, code=code) for code
                    in circuit_codes]

    plotted_codenames = set()
    for key in loss_record.keys():
        circuit, code, index = key
        show_label = False
        junction_count, inductor_count, _ = get_element_counts(circuit)
        # codename = lookup_codename(junction_count, inductor_count)
        if code not in plotted_codenames:
            show_label = True
            plotted_codenames.add(code)
        plot_circuit_metrics(circuit, loss_record, code, optimal_keys,
                             show_label=show_label)

    # plt.savefig('/home/mckeehan/sqcircuit/Qubit-Discovery/results/output.png')
    plt.savefig(f'/home/groups/safavi/sqcircuit/Qubit-Discovery/results/{type}_record.png')

def main():
    circuit_codes = [code for code in args.codes.split(',')]
    num_runs = int(args.num_runs)
    aggregate_loss_record = {}
    aggregate_metric_record = {}
    for codename in circuit_codes:
        for id in range(1, num_runs+1):
            loss_record = load_record(f'/home/groups/safavi/sqcircuit/Qubit-Discovery/results/loss_record_{codename}_{id}.pickle')
            metric_record = load_record(
                f'/home/groups/safavi/sqcircuit/Qubit-Discovery/results/metric_record_{codename}_{id}.pickle')
            aggregate_loss_record.update(loss_record)
            aggregate_metric_record.update(metric_record)
    plot_results(aggregate_loss_record, circuit_codes, type='loss')
    plot_results(aggregate_metric_record, circuit_codes, type='metrics')


if __name__ == "__main__":
    main()
