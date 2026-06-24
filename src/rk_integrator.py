import numpy as np
from tqdm import tqdm


def increment_state(
    c: np.ndarray,
    b: np.ndarray,
    mode_omegas: np.ndarray,
    atom_frequencies: np.ndarray,
    mode_couplings: np.ndarray,
    dt: float,
    t: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one interaction-picture Schrodinger increment in the one-photon sector.

    c[n] is the amplitude of one photon in mode n and all atoms in the ground
    state. b[j] is the amplitude of atom j excited and no photon.
    """

    detunings = mode_omegas[:, None] - atom_frequencies[None, :]
    interaction = mode_couplings * np.exp(-1j * detunings * t)

    c_increment = interaction.conj() @ b
    b_increment = c @ interaction

    return -1j * dt * c_increment, -1j * dt * b_increment


def rk_propagator(
    c_init: np.ndarray,
    b_init: np.ndarray,
    mode_omegas: np.ndarray,
    atom_frequencies: np.ndarray,
    mode_couplings: np.ndarray,
    time: dict[str, float],
    progress: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Propagate a one-photon state with a fourth-order Runge-Kutta scheme.
    """

    dt = float(time["dt"])
    n_steps = int(round(float(time["T"]) / dt)) + 1

    c_array = np.zeros((n_steps, len(c_init)), dtype=complex)
    b_array = np.zeros((n_steps, len(b_init)), dtype=complex)
    c_array[0] = c_init
    b_array[0] = b_init

    for i in tqdm(range(1, n_steps), disable=not progress):
        t = (i - 1) * dt
        c_current = c_array[i - 1]
        b_current = b_array[i - 1]

        c_1, b_1 = increment_state(
            c_current,
            b_current,
            mode_omegas,
            atom_frequencies,
            mode_couplings,
            dt,
            t,
        )
        c_2, b_2 = increment_state(
            c_current + c_1 / 2,
            b_current + b_1 / 2,
            mode_omegas,
            atom_frequencies,
            mode_couplings,
            dt,
            t + dt / 2,
        )
        c_3, b_3 = increment_state(
            c_current + c_2 / 2,
            b_current + b_2 / 2,
            mode_omegas,
            atom_frequencies,
            mode_couplings,
            dt,
            t + dt / 2,
        )
        c_4, b_4 = increment_state(
            c_current + c_3,
            b_current + b_3,
            mode_omegas,
            atom_frequencies,
            mode_couplings,
            dt,
            t + dt,
        )

        c_array[i] = c_current + (c_1 + 2 * c_2 + 2 * c_3 + c_4) / 6
        b_array[i] = b_current + (b_1 + 2 * b_2 + 2 * b_3 + b_4) / 6

    return c_array, b_array
