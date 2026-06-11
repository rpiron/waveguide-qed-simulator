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

def run_reflection_vs_RKstep(param_photon, param_cavity, T, dt_tab, cutoffs, 
                             index_experiment=0, index_omega_p = 0, store_results:bool=True, progress:bool=True):
    
    """
    Docstring for run_reflection_vs_RKstep
    
    """

    #prepare the array to store the results
    data_rows = []
    omega_0, gamma_0 = get_bare_param(param_cavity['omega_A'], param_cavity['gamma_A'], cutoffs['ir_cutoff'], cutoffs['uv_cutoff'], apply_corrections=True)
    param_cavity_simulation = {'omega_0': omega_0, 'gamma_0': gamma_0, 'L': param_cavity['L']}

    for i in tqdm(range(len(dt_tab)), disable=not progress):
        dt = dt_tab[i]
        param_time_evol = {'T': T, 'dt': dt}
        
        config = ExperimentConfig(param_photon=param_photon,
                                  param_cavity=param_cavity_simulation,
                                  param_time_evol=param_time_evol,
                                  cutoffs=cutoffs)
            
        experiment = Experiment(config)
        experiment.propagate_state(progress=False)
        _, _, Rn_array = experiment.compute_observables()
            
        data_rows.append({'dt': dt, 'final_reflection': Rn_array[-1]})
    
    if store_results:

        df = pd.DataFrame(data_rows)
        filename = (f'reflection_vs_RKstep_omegap_{index_omega_p}_xp{index_experiment}.csv' if index_experiment else 'reflection_vs_RKstep.csv')
        df.to_csv( project_root / 'single_photon_renormalization' / 'results' / 'csv_files' / 'reflection_vs_RKstep' / filename, index=False)

    return data_rows