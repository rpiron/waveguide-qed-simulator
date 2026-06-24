# Waveguide QED Simulator

## Overview

This repository contains a minimal momentum-space simulator for one-photon
light-matter scattering. The core code in `src/` now supports:

- one or two spatial dimensions;
- arbitrary periodic box lengths `L_1, ..., L_d`;
- one or several atoms with positions `r_j`, bare frequencies `Omega_j`, and
  continuum couplings `g_j`;
- vacuum dispersion `omega_p = ||p||`;
- RK4 propagation in the single-excitation sector.

The `benchmark_model/` folder keeps the one-dimensional single-atom benchmark
used in `main_tqe.pdf`, including its cutoff-renormalization helper and the
existing script/notebook/results workflow.

## Project Structure

```text
.
├── src/
│   ├── experiment.py       # Experiment class: modes, states, observables
│   ├── rk_integrator.py    # RK4 propagation in the one-photon sector
│   └── xp_config.py        # ExperimentConfig dataclass
├── benchmark_model/
│   ├── scripts_experiments/
│   ├── notebooks/
│   └── results/
├── 2d_setup/               # 2D fiber geometry experiments
│   ├── geometries/          # Saved atomic-interface CSV geometries
│   ├── notebooks/
│   └── results/
├── timedomain_cross_validation/
├── main_tqe.pdf
└── environment.yml
```

## Minimal Usage

```python
import numpy as np

from src.xp_config import ExperimentConfig
from src.experiment import Experiment

Omega = 10 * np.pi
g = np.sqrt(np.pi / 2)

config = ExperimentConfig(
    dimension=1,
    lengths=[50],
    atom_positions=[0.0],
    atom_frequencies=[Omega],
    atom_couplings=[1j * g],
    photon={"omega_p": Omega, "delta_k": 0.05 * np.pi, "x_0": -12.5},
    time={"T": 25, "dt": 0.01},
    cutoffs={"ir_cutoff": 7 * np.pi, "uv_cutoff": 13 * np.pi},
)

experiment = Experiment(config)
experiment.propagate_state(progress=True)
atom_pop, transmitted, reflected = experiment.compute_observables()
```

For 2D, use `dimension=2`, two box lengths, 2D atom positions, and a photon
direction:

```python
photon = {
    "omega_p": Omega,
    "delta_k": 0.2,
    "position": [-10, 0],
    "direction": [1, 0],
}
```

## Benchmark Workflow

The benchmark scripts keep their historical public signatures and can still be
driven from notebooks:

```python
from benchmark_model.scripts_experiments.reflection_vs_frequency import (
    run_reflection_vs_frequency,
)
```

Generated CSV files are stored under `benchmark_model/results/csv_files/`.

## 2D Fiber Setup

Use `2d_setup/notebooks/geometry_creation.ipynb` to generate and save a matter
interface CSV, then load it from
`2d_setup/notebooks/experiment_class_example_2d.ipynb` before choosing the bath,
wavepacket, and RK parameters.

## Environment

The simulations were developed for the Conda environment described in
`environment.yml`.
