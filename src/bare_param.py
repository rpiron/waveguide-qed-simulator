import numpy as np

def get_bare_param(omega_A, Gamma, omega_ref, lbda, precision=1e-4):
    """
    Computes the bare parameters by inverting the renormalization relations
    (omega_0, gamma) = F(omega_A, Gamma, omega_ref, lbda)
    
    Parameters:
    omega_A : physical transition frequency of the TLS
    Gamma : physical decay rate of the TLS
    omega_ref : central fequency of the numerical frequency window
    lbda : bandwith of the frequency window, defined as [IR, UV] = [omega_ref - lbda, omega_ref + lbda] 
    precision : precision parameter for the dichotomy research

    Returns:
    omega_0 : bare frequency to parameterize the Hamiltonian
    gamma : bare decay rate to parameterize the Hamiltonian
    """
    
    ## Obtain the bare frequency

    #deviation from the central frequency
    dwA = omega_A - omega_ref

    #Initial guess : omega0 in [omega_ref - lbda, omega_ref + lbda] (equivalently, dw0 in [-lbda, lbda])
    dw0_inf = -1*lbda 
    dw0_sup = lbda
    residual_diff = np.inf

    while np.abs(residual_diff) > precision: #research by dichotomy

        dw0_guess = 0.5*(dw0_inf + dw0_sup)
        residual_diff = dwA - dw0_guess + Gamma /(2*np.pi) * np.log((lbda-dw0_guess)/(lbda+dw0_guess)) 
        if residual_diff > 0:
            dw0_inf = dw0_guess
        else:
            dw0_sup = dw0_guess

    omega_0 = omega_ref + dw0_guess

    ## Obtain the bare decay rate
    gamma = 1/(1/Gamma + lbda / (np.pi*(lbda**2 - dw0_guess**2)))

    return omega_0, gamma