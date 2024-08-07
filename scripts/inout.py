"""This module provides functions to input and output of the scripts.

################################################################################
YAML Configuration File Structure(The yaml file is the core setting
for all the scripts):

- name: String. A unique identifier for the optimization setup.
- K: Integer. Total truncation number for circuit quantization.
- epochs: Integer. Total number of optimization iteration.
- num_eigenvalues: Integer. Total number of eigenvalues for diagonalization.
- capacitor_range: List of floats. Range of capacitor values in
- Farads, specified as [min, max].
- inductor_range: List of floats. Range of inductor values in
  Henry, specified as [min, max].
- junction_range: List of floats. Range of junction frequencies in
  radians per second, specified as [min, max].
- use_losses: Dictionary. Specifies which loss to use with float values for the
  weight in the total loss function:
    - frequency: Frequency loss.
    - anharmonicity: Anharmonicity loss.
    - flux_sensitivity: Flux sensitivity loss.
    - charge_sensitivity: Charge sensitivity loss.
    - T1: Energy relaxation time (T1) loss.
- use_metrics: List of strings. Metrics to use in simulations, e.g., ["T1"] for
  coherence time.

Example:
    name: "JL"
    K: 120
    epochs: 100
    capacitor_range: [1e-15, 12e-12]
    inductor_range: [1e-15, 5e-6]
    junction_range: [1e9 * 2 * np.pi, 100e9 * 2 * np.pi]
    num_eigenvalues: 10
    use_losses:
      frequency: 1.0
      anharmonicity: 1.0
      flux_sensitivity: 1.0
      charge_sensitivity: 1.0
      T1: 0.0
    use_metrics: ["T1"]
    optim_type: SGD
    circuit_code: JL
    init_circuit: ""

Ensure the YAML file follows this structure for proper processing.

################################################################################
The working tree for the outputs of scripts will look like:

 main_folder/                       - main directory
 ├── yaml_file.yaml
 │
 └── {optim_type}_{name}/           - experiment_directory
     │
     ├── records/                   - records_directory
     │   │
     │   ├── {optim_type}_loss_record_{circuit_code}_{name}_{seed}.pickle
     │   ├── {optim_type}_metrics_record_{circuit_code}_{name}_{seed}.pickle
     │   └── {optim_type}_circuits_record_{circuit_code}_{name}_{seed}.pickle
     │
     ├── plots/                     - plots_directory
     │   │
     │   ├── {circuit_code}_n_{num_runs}_{optim_type}_{name}_loss.png
     │   └── {circuit_code}_n_{num_runs}_{optim_type}_{name}_metrics.png
     │
     └── summaries/                 - summaries_directory
         │
         └── {optim_type}_circuit_summary_{circuit_code}_{name}_{id_num}.txt

Ensure that you have the correct file structure for proper operation
of the scripts and modules within this project.
"""

from collections import defaultdict
import os
from typing import List, Tuple, Dict

import yaml

from qubit_discovery.losses.loss import get_all_metrics

################################################################################
# General Settings.
################################################################################

# Keys that must be included in Yaml file.
YAML_KEYS = [
    'name',
    'K',
    "epochs",
    "num_eigenvalues",
    "capacitor_range",
    "inductor_range",
    "junction_range",
    "use_losses",
    "use_metrics",
]

# unit keys for metrics.
UNITS = {
    'frequency': '[GHz]',
    'gate_speed': '[GHz]',
    't2': '[s]',
    't1': '[s]',
    't1_capacitive': '[s]',
    't1_inductive': '[s]',
    't1_quasiparticle': '[s]',
    't_phi': '[s]',
    't_phi_charge': '[s]',
    't_phi_cc': '[s]',
    't_phi_flux': '[s]',
}
UNITS = defaultdict(lambda: "", UNITS)

################################################################################
# Read Functionalities.
################################################################################


def get_metrics_dict(config: dict) -> Tuple[List[str], List[str]]:
    """Return a list of the metrics that were not used in optimization and the
    metrics that were used in optimization.

    Parameters
    ----------
        config:
            A dictionary containing the parameters of the yaml file.
    """

    metrics_in_optim = list(config['use_losses'].keys())
    metrics_not_in_optim = []
    all_metrics = metrics_in_optim + config['use_metrics']

    for metric in all_metrics:
        if metric not in metrics_in_optim:
            metrics_not_in_optim.append(metric)

    return metrics_in_optim, metrics_not_in_optim


def get_units() -> Dict[str, str]:
    """Return a dictionary containing the units of the metrics"""

    return UNITS


def load_yaml_file(path_to_yaml: str) -> dict:
    """Load yaml file and assert that yaml file has the expected keys.

    Parameters
    ----------
        path_to_yaml:
            A string containing the path to the yaml file.
     """

    with open(path_to_yaml, 'r') as f:
        parameters = yaml.safe_load(f.read())

    for key in YAML_KEYS:
        assert key in parameters.keys(), (
            f'Yaml file must include \'{key}\' key.'
        )

    return parameters


def add_command_line_keys(
    parameters: dict,
    arguments: dict,
    keys: List[str],
    optional_keys: List[str] = None,
) -> dict:
    """Add command line argument keys to the parameters which are loaded from
    YAML file.

    Parameters
    ----------
        parameters:
            A dictionary containing the parameters of the YAML file.
        arguments:
            A dictionary containing the arguments of the command line.
        keys:
            A list of string keys that must be either specified in the YAML
            file or command line options.
        optional_keys:
            A list of string keys that are optional and can be not specified in
            both the YAML file and the command line options.
    """

    if optional_keys is None:
        optional_keys = []

    for key in keys + optional_keys:
        if arguments['--' + key] is not None or key in optional_keys:
            parameters[key] = arguments['--' + key]

        assert key in parameters.keys(), (
            f'\'{key}\' key must be either passed '
            f'in the command line or yaml file'
        )

    return parameters

################################################################################
# Directories
################################################################################


class Directory:
    """Directory class to address the files in the main directories.

    Parameters
    ----------
        parameters:
            A dictionary containing the parameters of YAML file/command line.
        yaml_filepath:
            The location of the YAML file.
    """

    def __init__(self, parameters: dict, yaml_filepath: str) -> None:

        self.parameters = parameters
        self.loc = os.path.dirname(os.path.abspath(yaml_filepath))

    def get_experiment_dir(self) -> str:
        """Return the directory for the experiment folder. The experiment folder
        has the following format:
            ``{name}/``
        """

        experiment_dir = os.path.join(
            self.loc,
            self.parameters['name']
        )

        # create the folder if if does not exist.
        os.makedirs(experiment_dir, exist_ok=True)

        return experiment_dir

    def get_records_dir(self) -> str:
        """Return the directory for the records folder inside the experiment
        folder.
        """

        records_dir = os.path.join(
            self.get_experiment_dir(),
            'records',
        )

        # create the folder if it's not excited.
        os.makedirs(records_dir, exist_ok=True)

        return records_dir

    def get_plots_dir(self) -> str:
        """Return the directory for the plot folder inside the experiment
        folder.
        """

        plots_dir = os.path.join(
            self.get_experiment_dir(),
            'plots'
        )

        # create the folder if it does not exist
        os.makedirs(plots_dir, exist_ok=True)

        return plots_dir

    def get_summaries_dir(self) -> str:
        """Return the directory for the summaries folder inside the experiment
        folder.
        """

        summaries_dir = os.path.join(
            self.get_experiment_dir(),
            'summaries'
        )

        # create the folder if it does not exist
        os.makedirs(summaries_dir, exist_ok=True)

        return summaries_dir

    def get_record_file_dir(
        self,
        record_type: str,
        circuit_code: str,
        idx: int,
    ) -> str:
        """Return the directory for the saved circuit record as pickled file
        in the records_directory.

        Parameters
        ----------
            record_type :
                A string indicating the type of record. It can be either 'loss',
                `metrics` or circuit.
            circuit_code:
                A string specifying circuit_code of the circuit.
            idx:
                An Integer specifying the index of the circuit in the records.
        """

        record_name = (
            f'{record_type}_record_{circuit_code}_{self.parameters["name"]}'
            + f'_{self.parameters["optim_type"]}_{idx}.pickle'
        )
        record_name = ''.join([str(word) for word in record_name])

        return os.path.join(
            self.get_records_dir(),
            record_name,
        )

    def get_summary_file_dir(self, circuit_code: str, idx: int) -> str:
        """Return the text file directory to save summary files inside summary
        folder.

        Parameters
        ----------
            circuit_code:
                A string specifying circuit_code of the circuit.
            idx:
                An integer specifying the index of the circuit in the records.
        """
        summary_name = (
            f'circuit_summary_{circuit_code}_{self.parameters["name"]}'
            + f'_{self.parameters["optim_type"]}_{idx}.txt'
        )

        return os.path.join(
            self.get_summaries_dir(),
            summary_name,
        )
