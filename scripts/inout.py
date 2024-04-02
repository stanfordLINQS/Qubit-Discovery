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
- use_metrics: List of strings. Metrics to use in simulations, e.g., ["T2"] for
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
    use_metrics: ["T2"]
    optim_type: SGD
    circuit_code: JL
    init_circuit: ""

Ensure the YAML file follows this structure for proper processing.

################################################################################
The working tree for the outputs of scripts will look like:

 main_folder/                       - main directory
 ├── yaml_file.yaml
 │
 ├── {optim_type}_{name}/           - experiment_directory
 │   │
 │   ├── records/                   - record_directory
 │   │   │
 │   │   ├── {optim_type}_loss_record_{circuit_code}_{name}_{seed}.pickle
 │   │   ├── {optim_type}_metrics_record_{circuit_code}_{name}_{seed}.pickle
 │   │   └── {optim_type}_circuits_record_{circuit_code}_{name}_{seed}.pickle
 │   │
 │   └── plots/                     - plots_directory
 │       │
 │       ├── {circuit_code}_n_{num_runs}_{optim_type}_{name}_loss.pickle
 │       └── {circuit_code}_n_{num_runs}_{optim_type}_{name}_metrics.pickle
 │
 └── experiment_folder_2/

Ensure that you have the correct file structure for proper operation
of the scripts and modules within this project.
"""
import yaml

from typing import List
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

################################################################################
# Read Functionalities.
################################################################################


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
            f"Yaml file must include \"{key}\" key."
        )

    return parameters


def add_command_line_keys(
    parameters: dict,
    arguments: dict,
    keys: List[str],
    optional_keys: List[str] = None,
) -> dict:
    """Add command line argument keys to the parameters which is loaded from
    yaml file.

    Parameters
    ----------
    parameters:
        A dictionary containing the parameters of the yaml file.
    arguments:
        A dictionary containing the arguments of the command line.
    keys:
        A list of string keys that must be either specified in the yaml
        file or command line options.
    optional_keys:
        A list of string keys that are optional and can be not specified in both
        the yaml file and the command line options.
    """

    if optional_keys is None:
        optional_keys = []

    for key in keys + optional_keys:
        if arguments['--' + key] is not None or key in optional_keys:
            parameters[key] = arguments['--' + key]

        assert key in parameters.keys(), (
            f"\"{key}\" key must be either passed "
            f"in the command line or yaml file"
        )

    return parameters

################################################################################
# Write Functionalities.
################################################################################
