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
from src.bare_param import get_bare_param_n

pi = np.pi

def run_reflection_vs_n(param_photon, param_cavity_physical, param_time_evol, cutoffs, n_tab,
                         index_omega_q = 0, index_experiment=0, store_results:bool=True, progress:bool=True):
    """
    To complete
    """

    reflection_tab = np.zeros(len(n_tab))

    for i in tqdm(range(len(n_tab)), disable=not progress):

        #Try to get the bare parameters
        try:
            omega_0, gamma = get_bare_param_n(param_cavity_physical['omega_A'], 
                                              param_cavity_physical['Gamma'], 
                                              cutoffs['ir_cutoff'], 
                                              cutoffs['uv_cutoff'], 
                                              n=n_tab[i])

            #Parameters of the simulation
            param_cavity = {'omega_0': omega_0, 'gamma': gamma, 'L': param_cavity_physical['L']}

            #Run the scattering experiment
            config = ExperimentConfig(param_photon=param_photon,
                                      param_cavity=param_cavity,
                                      param_time_evol=param_time_evol,
                                      cutoffs=cutoffs)
                
            experiment = Experiment(config)
            experiment.propagate_state(progress=False)

            #Compute the reflection
            _, _, Rn_array = experiment.compute_observables()

            reflection_tab[i] = Rn_array[-1]

            del experiment

        except Exception:
            print("WARNING : Bare parameters not found. Returning NaN")
            reflection_tab[i] = np.nan

    if store_results:
        data_to_save = {'n_tab': n_tab, 'reflection_tab': reflection_tab}
        df = pd.DataFrame(data_to_save)
        df.to_csv(project_root / 'single_photon_renormalization' /'results' / 'csv_files' / f'coincidence_vs_n_omega_{index_omega_q}_window_{index_experiment}.csv', index=False)
    
    return n_tab, reflection_tab