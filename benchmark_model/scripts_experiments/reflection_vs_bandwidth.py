import sys
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

#Local imports
from src.experiment import Experiment
from benchmark_model.scripts_experiments.renormalization import make_benchmark_config

pi = np.pi

def run_reflection_vs_bandwidth(param_photon, param_atom, param_time_evol, ir_tab, uv_tab, 
                                index_omega_p = 0, index_experiment = 0, correction:bool=False, store_results:bool=True, progress:bool=True):
    """
    Run the benchmark reflection experiment while varying the retained bandwidth.

    param_atom must contain {'Omega_A', 'g_A', 'L'}.
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
            config = make_benchmark_config(
                param_photon,
                param_atom,
                param_time_evol,
                cutoffs,
                correction=correction,
            )
                
            experiment = Experiment(config)
            experiment.propagate_state(progress=False)
            _, _, Rn_array = experiment.compute_observables()            
            reflection_tab[i] = Rn_array[-1]

            del experiment

    if store_results:
        data_to_save = {'ir_tab': ir_tab, 'uv_tab': uv_tab, 'reflection_tab': reflection_tab}
        df = pd.DataFrame(data_to_save)
        output_dir = project_root / 'benchmark_model' / 'results' / 'csv_files' / 'reflection_vs_bandwidth'
        output_dir.mkdir(parents=True, exist_ok=True)
        if correction:
            df.to_csv(output_dir / f'reflection_vs_bandwidth_omega{index_omega_p}_xp{index_experiment}_corrected.csv', index=False)
        else:
            df.to_csv(output_dir / f'reflection_vs_bandwidth_omega{index_omega_p}_xp{index_experiment}_uncorrected.csv', index=False)
        
    return ir_tab, uv_tab, reflection_tab
