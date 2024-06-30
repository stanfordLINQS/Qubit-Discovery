<br />
<p align="center">
<img src = pics/README_logo.png width= 500px" />
</p>

# Qubit Discovery
[**Introduction**](#Introduction)
|[**Tutorial**](#Tutorial)
|[**Installation**](#Installation)
|[**Examples**](#Examples)
|[**Contribution**](#Contribution)

## Introduction

Qubit-Discovery is a set of software tools to perform gradient-based optimization of superconducting qubits using the [`SQcircuit`](https://github.com/stanfordLINQS/SQcircuit) Python library. It provides a Python package `qubit_discovery` implementing loss functions and optimization algorithms, and a set of example scripts using them. 

With these tools, superconducting qubits can be optimized over the values of their elements for metrics such as decoherence time, anharmonicity, qubit frequency, or other user-defined targets.

A description of the theory and example usage is provided in the following paper:
> TBD

## Tutorial

### `qubit_discovery`

The `qubit_discovery` module provides two submodules:
- `qubit_discovery.optimization` has two optimization algorithms (SGD and BFGS), a sampler to randomly sample qubits, and a set of utilities to choose truncation numbers.
- `qubit_discovery.losses` has a set of functions to evaluate qubit metrics (anharmonicity, decoherence time, …) and utilties to build loss functions out of these metrics.

An example notebook demonstrating the usage on a single transmon can be found in `examples/tutorial.ipynb`.

### Scripts

The `scripts` directory provides a set of programs which can be used in a pipeline for parallel optimization of a superconducting circuit.
1. Run the `optimize.py` script many times in parallel to optimize from different initialization points.
2. Collate the results of the different runs with `plot_records.py`.
3. Output the details of the best-performing circuits using `circuit_summary.py`.

Each program takes in a common YAML file with the metadata about the optimization (circuit topology, allowed element ranges, …). An example is provided in `scripts/example_metadata.yaml`.

## Installation

To install the `qubit_discovery` package, both Conda and PyPI are supported:
```
pip install qubit_discovery
```
```
conda install -c conda-forge qubit_discovery
```

To use the provided scripts, after installing the `qubit_discovery` package clone the `Qubit-Discovery` repository and run with Python >=3.9. 

## Examples

Example notebooks demonstrating the use of the `qubit_discovery` package can be found in the `examples/` directory.

## Contribution


