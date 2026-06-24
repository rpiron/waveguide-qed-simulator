import numpy as np

from src.xp_config import ExperimentConfig

pi = np.pi


def get_bare_parameters(
    Omega_A: float,
    g_A: float,
    ir: float,
    uv: float,
    precision: float = 1e-4,
    apply_corrections: bool = False,
) -> tuple[float, float]:
    """
    Invert the benchmark-model cutoff corrections for Omega and g.

    The renormalization relation is written for g^2. The returned parameters
    are the bare Omega and positive coupling amplitude g used by the simulator.
    """

    if not apply_corrections:
        return Omega_A, g_A

    g_A_squared = float(g_A) ** 2

    Omega_inf = 1.01 * ir
    Omega_sup = 0.99 * uv
    residual_diff = np.inf
    n_iter = 0

    while abs(residual_diff) > precision and n_iter < 1e4:
        Omega_guess = 0.5 * (Omega_inf + Omega_sup)
        residual_diff = (
            Omega_A
            - Omega_guess
            + g_A_squared / pi * np.log((uv - Omega_guess) / (Omega_guess - ir))
        )

        if residual_diff > 0:
            Omega_inf = Omega_guess
        else:
            Omega_sup = Omega_guess
        n_iter += 1

    g_squared = g_A_squared / (
        1 - g_A_squared / pi * (1 / (ir - Omega_guess) - 1 / (uv - Omega_guess))
    )

    if n_iter < 1e4:
        return Omega_guess, np.sqrt(g_squared)
    return np.nan, np.nan


def make_benchmark_config(
    param_photon: dict[str, float],
    param_atom: dict[str, float],
    param_time_evol: dict[str, float],
    cutoffs: dict[str, float],
    correction: bool = False,
) -> ExperimentConfig:
    if "Omega_A" not in param_atom or "g_A" not in param_atom:
        raise ValueError("param_atom must define Omega_A, g_A and L.")

    Omega, g = get_bare_parameters(
        param_atom["Omega_A"],
        param_atom["g_A"],
        cutoffs["ir_cutoff"],
        cutoffs["uv_cutoff"],
        apply_corrections=correction,
    )

    return ExperimentConfig(
        dimension=1,
        lengths=[param_atom["L"]],
        atom_positions=[0.0],
        atom_frequencies=[Omega],
        atom_couplings=[1j * g],
        photon=param_photon,
        time=param_time_evol,
        cutoffs=cutoffs,
    )
