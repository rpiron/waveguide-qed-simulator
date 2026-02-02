# Waveguide QED Simulator

This repository contains the code used to generate the results presented in the paper  
**Renormalization Treatment of IR and UV Cutoffs in Waveguide QED and Implications to Numerical Model Simulation**.

The core of the project is the `Experiment` class, which simulates single-photon scattering events on a two-level system (TLS).  
Experiments are primarily driven from Jupyter notebooks, which are used both to generate the data and to visualize the results.

Users may either reproduce the simulations or load the pre-generated data stored in CSV files and reproduce the figures directly from the notebooks.



## Project structure

```text
.
├── src/
│   ├── experiment.py                 # Definition of the Experiment class
│   ├── rg_integrator.py              # Runge–Kutta integrator for the Schrödinger equation
│   └── xp_config.py                  # Configuration associated with the Experiment class
├── scripts_experiments/
│   ├── reflection_convergence.py     # Reflection coefficient at bare resonance vs UV cutoff
│   ├── reflection_vs_frequency.py    # Reflection vs photon frequency for different frequency windows
│   └── reflection_vs_rgstep.py       # Convergence of reflection vs number of Runge–Kutta time steps
├── single_photon_renormalization/
│   ├── notebooks/
│   │   ├── experiment_class_example.ipynb          # Results for Fig. 2
│   │   ├── reflection_convergence_results.ipynb    # Results for Fig. 3
│   │   ├── reflection_vs_frequency_results.ipynb   # Results for Figs. 4 and 5
│   │   └── reflection_vs_rgstep_results.ipynb      # Results for Fig. 7
│   └── results/
├── environment.yml
└── README.md

```

## Associated paper
[arXiv:2601.15945](https://arxiv.org/abs/2601.15945)


