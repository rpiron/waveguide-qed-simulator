import numpy as np
from scipy.optimize import root
from sympy import gamma

pi = np.pi


def get_bare_param(omega_A, gamma_A, ir, uv, precision=1e-4, apply_corrections:bool=False):
    """
    Computes the bare parameters by inverting the renormalization relations
    (omega_0, gamma) = F(omega_A, gamma_A, omega_ref, lbda)
    
    Parameters:
    omega_A : physical transition frequency of the TLS
    gamma_A : physical decay rate of the TLS
    ir: infrared cutoff of the spectral density
    uv: ultraviolet cutoff of the spectral density
    precision : precision parameter for the dichotomy research
    apply_corrections : whether to apply corrections to the bare parameters

    Returns:
    omega_0 : bare frequency to parameterize the Hamiltonian
    gamma : bare decay rate to parameterize the Hamiltonian
    """
    
    if not apply_corrections:
        return omega_A, gamma_A
    else:

        ## Obtain the bare frequenc by a dichtomy search
        w0_inf = 1.01*ir
        w0_sup = 0.99*uv
        residual_diff = np.inf
        N_iter = 0

        while np.abs(residual_diff) > precision and N_iter < 1e4: #research by dichotomy

            w0_guess = 0.5*(w0_inf + w0_sup)
            residual_diff = omega_A - w0_guess + gamma_A /(2*pi) * np.log((uv - w0_guess) / (w0_guess - ir))

            if residual_diff > 0:
                w0_inf = w0_guess
            else:
                w0_sup = w0_guess
            N_iter += 1

        #Deduce gamma accordingly
        gamma_0 = gamma_A / (1 - gamma_A/(2*pi)*(1/(ir - w0_guess) - 1/(uv - w0_guess)))
        
        if N_iter <= 1e4:
            return w0_guess, gamma_0
        else:
            return np.nan, np.nan
