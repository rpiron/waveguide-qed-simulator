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
from src.bare_param import get_bare_param

def run_reflection_vs_frequency(param_photon_bis, param_cavity, param_time_evol, frequency_values, cutoffs, 
                                index_experiment=0, correction:bool=False, store_results:bool=True, progress:bool=True):

    """
    Runs the reflection against frequency experiment for different photon central frequencies.
    
    Parameters:
    param_photon_bis (Dict) : Dictionary containing only {'delta_k', 'x_0'}
    param_cavity (Dict) : Dictionary containing {'omega_A', 'gamma_A', 'L'}
    param_time_evol (Dict) : Dictionary containing {'T', 'dt'}
    cutoffs (Dict) : Dictionary containing {'ir_cutoff', 'uv_cutoff'} 
    frequency_values (np.array) : Array of photon central frequency values
    index_experiment (int) : Index of the experiment if multiple are run in sequence.
    store_results (bool) : Whether to store the results in a CSV file.
    progress (bool) : Whether to display a progress bar.

    Returns:
    frequency_values (np.array) : Array of photon central frequency values used.
    final_reflection_tab (np.array) : Array of final reflection values corresponding to each frequency.

    """

    #prepare the array to store the results
    final_reflection_tab = np.zeros(len(frequency_values))

    for i in tqdm(range(len(frequency_values)), disable=not progress):
        omega_p = frequency_values[i]
        param_photon_current = param_photon_bis.copy()
        param_photon_current['omega_p'] = omega_p

        omega_0, gamma_0 = get_bare_param(param_cavity['omega_A'], param_cavity['gamma_A'], cutoffs['ir_cutoff'], cutoffs['uv_cutoff'], apply_corrections=correction)
        param_cavity_simulation = {'omega_0': omega_0, 'gamma_0': gamma_0, 'L': param_cavity['L']}

        config = ExperimentConfig(param_photon=param_photon_current,
                                  param_cavity=param_cavity_simulation,
                                  param_time_evol=param_time_evol,
                                  cutoffs=cutoffs)
        
        experiment = Experiment(config)
        experiment.propagate_state(progress=False)
        _, _, Rn_array = experiment.compute_observables()
        final_reflection_tab[i] = Rn_array[-1]
    
    if store_results:
        data_to_save = {'photon_frequency_tab': frequency_values, 'final_reflection_tab': final_reflection_tab}
        df = pd.DataFrame(data_to_save)
        if correction:
            df.to_csv(project_root / 'single_photon_renormalization' / 'results' / 'csv_files' /f'reflection_vs_frequency_{index_experiment}_corrected.csv', index=False)
        else:
            df.to_csv(project_root / 'single_photon_renormalization' / 'results' / 'csv_files' /f'reflection_vs_frequency_{index_experiment}_uncorrected.csv', index=False)

    return frequency_values, final_reflection_tab