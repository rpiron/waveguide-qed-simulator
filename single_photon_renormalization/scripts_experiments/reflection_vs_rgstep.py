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

def run_reflection_vs_rgstep(param_photon_bis, param_cavity, T, frequency_values, dt_tab, cutoffs, index_experiment=0,
                             store_results:bool=True, progress:bool=True):
    
    """
    Docstring for run_reflection_vs_rgstep
    
    """

    #prepare the array to store the results
    data_rows = []

    for i in tqdm(range(len(dt_tab)), disable=not progress):
        dt = dt_tab[i]
        param_time_evol = {'T': T, 'dt': dt}

        for j in range(len(frequency_values)):

            omega_p = frequency_values[j]
            param_photon_current = param_photon_bis.copy()
            param_photon_current['omega_p'] = omega_p

            config = ExperimentConfig(param_photon=param_photon_current,
                                      param_cavity=param_cavity,
                                      param_time_evol=param_time_evol,
                                      cutoffs=cutoffs)
            
            experiment = Experiment(config)
            experiment.propagate_state(progress=False)
            _, _, Rn_array = experiment.compute_observables()
            
            data_rows.append({'dt': dt, 'omega_p': omega_p, 'final_reflection': Rn_array[-1]})
    
    if store_results:

        df = pd.DataFrame(data_rows)
        filename = (f'reflection_vs_rgstep_{index_experiment}.csv' if index_experiment else 'reflection_vs_rgstep.csv')
        df.to_csv( project_root / 'single_photon_renormalization' / 'results' / 'csv_files' / filename, index=False)

    return data_rows