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

def run_reflection_vs_n(omega_q, ir_tab, uv_tab, index_omega_q = 0, n=1, 
                         store_results:bool=True, progress:bool=True):
    """
    To complete
    """

    #Parameterize the experiment

    omega_A = 10*pi
    Gamma = pi
    reflection_tab = np.zeros(len(ir_tab))

    for i in tqdm(range(len(ir_tab)), disable=not progress):

        #Frequency window
        cutoffs = {'ir_cutoff': ir_tab[i] , 'uv_cutoff': uv_tab[i]}

        #Sanity check
        if omega_q < cutoffs['ir_cutoff'] or omega_q > cutoffs['uv_cutoff'] :
            print("WARNING : The photon frequency is not included in the frequency window. Returning NaN")
            reflection_tab[i] = np.nan
        
        else:
            #Bare parameters
            try:
                omega_0, gamma = get_bare_param_n(omega_A, Gamma, ir_tab[i], uv_tab[i], n=n)

                #Parameters of the simulation
                L = 50

                param_cavity = {'omega_0': omega_0, 'gamma': gamma, 'L': L}

                param_time_evol = {'T': L/2, 'dt': 0.01}

                param_photon = {'omega_p': omega_q, 'delta_k': 0.05*np.pi, 'x_0': -L/4}  

                #Run the scattering experiment
                config = ExperimentConfig(param_photon=param_photon,
                                        param_cavity=param_cavity,
                                        param_time_evol=param_time_evol,
                                        cutoffs=cutoffs)
                
                experiment = Experiment(config)
                experiment.propagate_state(progress=False)

                #Compute the coindicence only at final time to save computational resources
                _, _, Rn_array = experiment.compute_observables()

                reflection_tab[i] = Rn_array[-1] 

                del experiment

            except Exception:
                print("WARNING : Bare parameters not found. Returning NaN")
                reflection_tab[i] = np.nan

    if store_results:
        data_to_save = {'ir_tab': ir_tab, 'uv_tab': uv_tab, 'reflection_tab': reflection_tab}
        df = pd.DataFrame(data_to_save)
        df.to_csv(project_root / 'single_photon_renormalization' / 'results' / 'csv_files' / f'reflection_vs_n{n}_ir{int(ir_tab[0]/pi)}_{index_omega_q}.csv', index=False)
    
    return ir_tab, uv_tab, reflection_tab