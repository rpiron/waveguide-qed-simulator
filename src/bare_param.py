import numpy as np
from scipy.optimize import root

pi = np.pi

def get_bare_param_n2(omega_A, Gamma, ir, uv, precision=1e-4):
    """
    Computes the bare parameters by inverting the renormalization relations
    (omega_0, gamma) = F(omega_A, Gamma, omega_ref, lbda)
    
    Parameters:
    omega_A : physical transition frequency of the TLS
    Gamma : physical decay rate of the TLS
    ir: infrared cutoff of the spectral density
    uv: ultraviolet cutoff of the spectral density
    precision : precision parameter for the dichotomy research

    Returns:
    omega_0 : bare frequency to parameterize the Hamiltonian
    gamma : bare decay rate to parameterize the Hamiltonian
    """
    
    ##Express the bare decay as a function of omega_0
    def adjust_gamma(x):
        return Gamma / (1 - Gamma/(2*pi) * ((ir + omega_A - 2*x)/(ir - x)**2 - (uv + omega_A - 2*x)/(uv - x)**2))

    ## Obtain the bare frequenc by a dichtomy search
    w0_inf = 1.01*ir
    w0_sup = 0.99*uv
    residual_diff = np.inf
    N_iter = 0

    while np.abs(residual_diff) > precision and N_iter < 1e4: #research by dichotomy

        w0_guess = 0.5*(w0_inf + w0_sup)
        gamma_guess = adjust_gamma(w0_guess)
        residual_diff = -1*gamma_guess/(4*pi) * (1/(ir - w0_guess)**2 - 1/(uv - w0_guess)**2) * (Gamma**2 / 4 - (w0_guess - omega_A)**2) \
                        - (1 + gamma_guess/(2*pi) * (1/(ir - w0_guess) - 1/(uv - w0_guess))) * (w0_guess - omega_A) \
                        + gamma_guess / (2*pi) * np.log((uv - w0_guess)/(w0_guess - ir))

        if residual_diff > 0:
            w0_inf = w0_guess
        else:
            w0_sup = w0_guess
        N_iter += 1

    #Last round needed to update gamma
    gamma_guess = adjust_gamma(w0_guess)

    if N_iter <= 1e4:
        return w0_guess, gamma_guess
    else:
        return np.nan, np.nan


def get_bare_param_n1(omega_A, Gamma, ir, uv, precision=1e-4):
    """
    Computes the bare parameters by inverting the renormalization relations
    (omega_0, gamma) = F(omega_A, Gamma, omega_ref, lbda)
    
    Parameters:
    omega_A : physical transition frequency of the TLS
    Gamma : physical decay rate of the TLS
    ir: infrared cutoff of the spectral density
    uv: ultraviolet cutoff of the spectral density
    precision : precision parameter for the dichotomy research

    Returns:
    omega_0 : bare frequency to parameterize the Hamiltonian
    gamma : bare decay rate to parameterize the Hamiltonian
    """
    
    ## Obtain the bare frequenc by a dichtomy search
    w0_inf = 1.01*ir
    w0_sup = 0.99*uv
    residual_diff = np.inf
    N_iter = 0

    while np.abs(residual_diff) > precision and N_iter < 1e4: #research by dichotomy

        w0_guess = 0.5*(w0_inf + w0_sup)
        residual_diff = omega_A - w0_guess + Gamma /(2*pi) * np.log((uv - w0_guess) / (w0_guess - ir))

        if residual_diff > 0:
            w0_inf = w0_guess
        else:
            w0_sup = w0_guess
        N_iter += 1

    #Deduce gamma accordingly
    gamma = Gamma / (1 - Gamma/(2*pi)*(1/(ir - w0_guess) - 1/(uv - w0_guess)))
    
    if N_iter <= 1e4:
        return w0_guess, gamma
    else:
        return np.nan, np.nan


def get_bare_param_n(omega_A, Gamma, ir, uv, n=1):
    """
    Computes the bare parameters by inverting the renormalization relations
    (omega_0, gamma) = F(omega_A, Gamma, omega_ref, lbda)
    
    Parameters:
    omega_A : physical transition frequency of the TLS
    Gamma : physical decay rate of the TLS
    ir: infrared cutoff of the spectral density
    uv: ultraviolet cutoff of the spectral density
    n : maximal n to keep in the alpha truncation

    Returns:
    omega_0 : bare frequency to parameterize the Hamiltonian
    gamma : bare decay rate to parameterize the Hamiltonian
    """

    #n=0 serves as a baseline : no correction in the bare parameters
    if n == 0:
        omega_0 =  omega_A
        gamma = Gamma

    else:
        def F(omega_0_guess, gamma_guess):
            #Store the alpha coeffcients
            polynom_tab = np.zeros(n+1, dtype=complex)

            X = (1j* (omega_0_guess - omega_A) - Gamma/2)

            polynom_tab[0] = -gamma_guess/2 + 1j*gamma_guess/(2*pi)*np.log(np.abs((uv-omega_0_guess)/(omega_0_guess - ir)))

            for i in range(1, n+1):
                #Runing dummy tests here : this is not the analytical experssion I've found
                polynom_tab[i] = (-1j)**(i-1) * gamma_guess / (2*i*pi) * \
                                ((omega_0_guess - ir)**(-i) + (-1)**(i-1) * (uv - omega_0_guess)**(-i)) \
                                * X**i

            error_term = np.sum(polynom_tab) - X

            return error_term
        
        def F_real(vars):
            omega_0, gamma = vars
            val = F(omega_0, gamma)
            return [val.real, val.imag]

        initial_guess = get_bare_param_n1(omega_A, Gamma, ir, uv)
        sol = root(F_real, initial_guess, tol=1e-10)
            
        omega_0, gamma = sol.x

    return omega_0, gamma


def get_bare_param(omega_A, Gamma, ir, uv):
    """
    Computes the bare parameters by inverting the renormalization relations
    (omega_0, gamma) = F(omega_A, Gamma, omega_ref, lbda)
    
    Parameters:
    omega_A : physical transition frequency of the TLS
    Gamma : physical decay rate of the TLS
    ir: infrared cutoff of the spectral density
    uv: ultraviolet cutoff of the spectral density

    Returns:
    omega_0 : bare frequency to parameterize the Hamiltonian
    gamma : bare decay rate to parameterize the Hamiltonian
    """
    def F_full(omega_0_guess, gamma_guess):
        #Store the alpha coeffcients
        
        X = 1j*(omega_0_guess - omega_A) - Gamma/2

        error_term = - gamma_guess/2 + 1j*gamma_guess/(2*pi) * np.log((uv - omega_0_guess)/(omega_0_guess - ir)) \
                    + 1j * gamma_guess/(2*pi) * np.log(1 - 1j*X/(uv - omega_0_guess)) \
                    - 1j * gamma_guess/(2*pi) * np.log(1 + 1j*X/(omega_0_guess - ir)) \
                    - X
                    
        return error_term

    def F_real_full(vars):
        omega_0, gamma = vars
        val = F_full(omega_0, gamma)
        return [val.real, val.imag]

    initial_guess = get_bare_param_n1(omega_A, Gamma, ir, uv)
    sol = root(F_real_full, initial_guess, tol=1e-10)
    omega_0, gamma = sol.x

    return omega_0, gamma