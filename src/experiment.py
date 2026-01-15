from src.rg_integrator import rg_propagator
from src.xp_config import ExperimentConfig
import numpy as np
#from src.ed_integrator import  # IGNORE

class Experiment:

    ###Initialize the experiment with the given configuration
    def __init__(self, config:ExperimentConfig):

        #integrator can be rg_propagator or ed_propagator (default: rg_propagator)
        self.integrator_func = config.integrator_func
        
        #parameters of the photon wavepacket and of the cavity
        self.param_photon = config.param_photon
        self.param_cavity = config.param_cavity
        self.param_time_evol = config.param_time_evol

        #ir and uv cutoffs for the frequency modes
        self.ir_cutoff = config.cutoffs['ir_cutoff']
        self.uv_cutoff = config.cutoffs['uv_cutoff']

        #Dictionnary to store potential error messages or warnings
        self.messages = {}

        #Frequency modes array assoiciated with the experiment
        omega_tab_half = np.array([2*np.pi*n/self.param_cavity['L'] for n in range(1000000) \
                                   if (2*np.pi*n/self.param_cavity['L'] <= self.uv_cutoff and \
                                       2*np.pi*n/self.param_cavity['L'] >= self.ir_cutoff)])
        
        self.omega_tab = np.concatenate((omega_tab_half, omega_tab_half))
        self.n_modes = len(omega_tab_half)

        #initialize arrays to store the time evolution of the state
        self.c_array = np.zeros((int(self.param_time_evol['T']/self.param_time_evol['dt']), 2*self.n_modes), dtype=complex)
        self.b_array =  np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=complex)

        #initialize arrays to store the excited state population and photon number at each time step
        self.An_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)
        self.Rn_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)
        self.Tn_array = np.zeros(int(self.param_time_evol['T']/self.param_time_evol['dt']), dtype=float)

        #test monochromatic limit
        if self.param_photon['delta_k'] / (self.param_cavity['gamma'] / 2) > 0.1:
            self.messages['monochromatic_limit'] = "Warning: The photon wavepacket is not in the monochromatic limit."
    
    def check_parameters(self):
        #to be completed: check that the config
        return True

    def propagate_state(self, progress:bool=False):

        #initialize the state
        b_init = 0
        c_init = np.exp(-(self.omega_tab - self.param_photon['omega_p'])**2 /(4*self.param_photon['delta_k']**2)) \
                 * np.exp(-1j * self.omega_tab * self.param_photon['x_0']) \
                    * np.concatenate((np.ones(self.n_modes), np.zeros(self.n_modes)))
        c_init = c_init / np.linalg.norm(c_init)

        #propagate the state using the selected integrator
        c_array, b_array = self.integrator_func(c_init, b_init, self.omega_tab, self.param_cavity, self.param_time_evol, progress=progress)

        self.c_array = c_array
        self.b_array = b_array

        return c_array, b_array
    
    def compute_observables(self):

        #compute the excited state population and photon number at each time step
        self.An_array = np.abs(self.b_array)**2
        self.Tn_array = np.sum(np.abs(self.c_array[:, :self.n_modes])**2, axis=1)
        self.Rn_array = np.sum(np.abs(self.c_array[:, self.n_modes:])**2, axis=1)

        #test probability conservation at final time
        if not np.isclose(self.Rn_array[-1] + self.Tn_array[-1] + self.An_array[-1], 1.0, atol=1e-3) :
            self.messages['probability_conservation'] = f"Error: Probability not conserved at final time: R + T + A = {self.Rn_array[-1] + self.Tn_array[-1] + self.An_array[-1]:.6f} != 1."

        #Check population of excited atomic state at final time
        if self.An_array[-1] > 1e-2:
            self.messages['final_excited_state'] = f"Warning: The population of the excited state at final time is {self.An_array[-1]:.2e}, which is significant."

        return self.An_array, self.Tn_array, self.Rn_array

    def get_messages(self):
        if self.messages:
            print("Current messsages:")
            for msg in self.messages:
                print("-", msg)
                print(self.messages[msg])

        return self.messages
        
