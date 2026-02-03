# Waveguide QED Simulator

## Overview

This repository contains the code used to generate the results presented in the paper  
**Renormalization Treatment of IR and UV Cutoffs in Waveguide QED and Implications to Numerical Model Simulation**.

The core of the project is the `Experiment` class, which simulates single-photon scattering events on a two-level system (TLS).  
Experiments are primarily driven from Jupyter notebooks, which are used both to generate the data and to visualize the results.

Users may either reproduce the simulations from scratch or load the pre-generated CSV data to reproduce the figures directly from the notebooks.
*All simulations were run using the Conda environment provided in `environment.yml`.*

## Project structure

```text
.
├── src/
│   ├── __init__.py
│   ├── experiment.py                  # Definition of the Experiment class
│   ├── rk_integrator.py               # Runge–Kutta integrator for the Schrödinger equation
│   └── experiment_config.py           # Configuration associated with the Experiment class
├── single_photon_renormalization/
│   ├── scripts_experiments/
│   │   ├── reflection_convergence.py      # Reflection coefficient at bare resonance vs UV cutoff
│   │   ├── reflection_vs_frequency.py     # Reflection vs photon frequency for different frequency windows
│   │   └── reflection_vs_rk_steps.py      # Convergence of reflection vs number of Runge–Kutta steps
│   ├── notebooks/
│   │   ├── experiment_class_example.ipynb          # Results for Fig. 2 
│   │   ├── reflection_convergence_results.ipynb    # Results for Fig. 3 
│   │   ├── reflection_vs_frequency_results.ipynb   # Results for Figs. 4 and 5 
│   │   └── reflection_vs_rgstep_results.ipynb      # Results for Fig. 7
│   ├── results/
│   │   ├── csv_files/   # Pre-generated CSV data from scripts_experiments
│   │   └── figures/     # Figures presented in the paper
├── environment.yml
└── README.md

```

## How to use

### 1. Clone the repository
```bash
git clone https://github.com/rpiron/waveguide-qed-simulator.git
cd waveguide-qed-simulator
```
### 2. Create the Conda environment
This will install all required dependencies (Python, NumPy, SciPy, Jupyter, etc.) in a reproducible environment.
```bash
conda env create -f environment.yml
conda activate single-photon-renorm
```
### 3. Launch Jupyter lab
```bash
jupyter lab
```
Then, navigate to
```bash
single_photon_renormalization/notebooks/
```

## Associated paper
Each notebook reproduces one or more figures of the associated paper.
[arXiv:2601.15945](https://arxiv.org/abs/2601.15945)


