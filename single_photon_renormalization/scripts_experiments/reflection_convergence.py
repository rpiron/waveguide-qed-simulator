import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

project_root = Path().resolve().parents[1]
sys.path.append(str(project_root))

#Local imports
from src.xp_config import ExperimentConfig
from src.experiment import Experiment


def run_reflection_convergence(param_photon, param_cavity, param_time_evol, uv_cutoff_values, store_results:bool=True, progress:bool=True):

    """
    Docstring for run_reflection_convergence
    
    :param param_photon: Description
    :param param_cavity: Description
    :param param_time_evol: Description
    :param uv_cutoff_values: Description
    """

    #prepare the array to store the results
    final_reflection_tab = np.zeros(len(uv_cutoff_values))

    for i in tqdm(range(len(uv_cutoff_values)), disable=not progress):
        uv_cutoff = uv_cutoff_values[i]
        cutoffs = {'ir_cutoff': 0, 'uv_cutoff': uv_cutoff}

        config = ExperimentConfig(param_photon=param_photon,
                                  param_cavity=param_cavity,
                                  param_time_evol=param_time_evol,
                                  cutoffs=cutoffs)
        
        experiment = Experiment(config)
        experiment.propagate_state(progress=False)
        _, _, Rn_array = experiment.compute_observables()
        final_reflection_tab[i] = Rn_array[-1]
    
    if store_results:
        data_to_save = {'uv_cutoff_tab': uv_cutoff_values, 'final_reflection_tab': final_reflection_tab}
        df = pd.DataFrame(data_to_save)
        df.to_csv(project_root / 'single_photon_renormalization' / 'results' / 'csv_files' /'reflection_convergence.csv', index=False)
    
    return uv_cutoff_values, final_reflection_tab

