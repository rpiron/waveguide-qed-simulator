from dataclasses import dataclass
from typing import Dict, Optional, Any
from src.rg_integrator import rg_propagator
import numpy as np

@dataclass
class ExperimentConfig:
    """
    Configuration class for Experiment parameters.
    """

    param_photon: Dict[str, float]
    param_cavity: Dict[str, float]
    param_time_evol: Dict[str, float]
    cutoffs: Dict[str, float]
    integrator_func: Optional[Any] = rg_propagator