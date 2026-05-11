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

pi = np.pi

def run_reflection_vs_bandwidth(param_photon, param_cavity_physical, param_time_evol, ir_tab, uv_tab, 
                                index_omega_p = 0, index_experiment = 0, correction:bool=False, store_results:bool=True, progress:bool=True):
    """
    To be completed
    """

    #Prepare to store the output
    reflection_tab = np.zeros(len(ir_tab))

    for i in tqdm(range(len(ir_tab)), disable=not progress):

        #Frequency window
        cutoffs = {'ir_cutoff': ir_tab[i] , 'uv_cutoff': uv_tab[i]}

        #Sanity check
        if param_photon['omega_p'] < cutoffs['ir_cutoff'] or param_photon['omega_p'] > cutoffs['uv_cutoff'] :
            print("WARNING : The photon frequency is not included in the frequency window. Returning NaN")
            reflection_tab[i] = np.nan
        
        else:
            #Bare parameters
            omega_0, gamma_0 = get_bare_param(param_cavity_physical['omega_A'], 
                                              param_cavity_physical['gamma_A'], 
                                              cutoffs['ir_cutoff'], 
                                              cutoffs['uv_cutoff'], apply_corrections=correction)
            
            param_cavity_simulation = {'omega_0': omega_0, 'gamma_0': gamma_0, 'L': param_cavity_physical['L']}

            config = ExperimentConfig(param_photon=param_photon,
                                    param_cavity=param_cavity_simulation,
                                    param_time_evol=param_time_evol,
                                    cutoffs=cutoffs)
                
            experiment = Experiment(config)
            experiment.propagate_state(progress=False)
            _, _, Rn_array = experiment.compute_observables()            
            reflection_tab[i] = Rn_array[-1]

            del experiment

    if store_results:
        data_to_save = {'ir_tab': ir_tab, 'uv_tab': uv_tab, 'reflection_tab': reflection_tab}
        df = pd.DataFrame(data_to_save)
        if correction:
            df.to_csv(project_root / 'single_photon_renormalization' / 'results' / 'csv_files' / 'reflection_vs_bandwidth' / f'reflection_vs_bandwidth_omega{index_omega_p}_xp{index_experiment}_corrected.csv', index=False)
        else:
            df.to_csv(project_root / 'single_photon_renormalization' / 'results' / 'csv_files' / 'reflection_vs_bandwidth' / f'reflection_vs_bandwidth_omega{index_omega_p}_xp{index_experiment}_uncorrected.csv', index=False)
        
    return ir_tab, uv_tab, reflection_tab