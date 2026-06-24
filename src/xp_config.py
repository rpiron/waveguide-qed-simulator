from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from src.rk_integrator import rk_propagator


def _as_1d_array(value: Any, dtype: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = array.reshape(1)
    return array.reshape(-1)


@dataclass
class ExperimentConfig:
    """
    Configuration for one-photon scattering in a 1D or 2D vacuum bath.

    The generic model is defined by:
    - dimension d in {1, 2};
    - periodic box lengths L_1, ..., L_d;
    - atom positions x_j, bare frequencies Omega_j and continuum couplings d_j;
    - a Gaussian incoming photon wavepacket.

    Couplings are the continuum amplitudes d_j in the Hamiltonian. For the
    one-dimensional benchmark convention g(p)=i*g, pass atom_couplings=[1j*g].
    """

    dimension: int
    lengths: Sequence[float]
    atom_positions: Sequence[Sequence[float]] | Sequence[float] | float
    atom_frequencies: Sequence[float] | float
    atom_couplings: Sequence[complex] | complex
    photon: Mapping[str, Any]
    time: Mapping[str, float]
    cutoffs: Mapping[str, float]
    integrator_func: Callable[..., tuple[np.ndarray, np.ndarray]] = rk_propagator

    def __post_init__(self) -> None:
        self.dimension = int(self.dimension)
        if self.dimension not in (1, 2):
            raise ValueError("Only dimension=1 or dimension=2 are supported.")

        self.lengths = _as_1d_array(self.lengths, float, "lengths")
        if len(self.lengths) != self.dimension:
            raise ValueError("lengths must contain one value per dimension.")
        if np.any(self.lengths <= 0):
            raise ValueError("All box lengths must be positive.")

        self.atom_frequencies = _as_1d_array(
            self.atom_frequencies, float, "atom_frequencies"
        )
        self.atom_couplings = _as_1d_array(
            self.atom_couplings, complex, "atom_couplings"
        )
        if len(self.atom_frequencies) != len(self.atom_couplings):
            raise ValueError("atom_frequencies and atom_couplings must have same length.")

        self.atom_positions = self._normalize_positions(self.atom_positions)
        if len(self.atom_positions) != len(self.atom_frequencies):
            raise ValueError("atom_positions must contain one position per atom.")

        self.photon = dict(self.photon)
        self.time = dict(self.time)
        self.cutoffs = self._normalize_cutoffs(dict(self.cutoffs))

        self._validate_photon()
        self._validate_time()

    @property
    def n_atoms(self) -> int:
        return len(self.atom_frequencies)

    @property
    def Omega_j(self) -> np.ndarray:
        return self.atom_frequencies

    @property
    def d_j(self) -> np.ndarray:
        return self.atom_couplings

    @property
    def volume(self) -> float:
        return float(np.prod(self.lengths))

    @property
    def omega_min(self) -> float:
        return self.cutoffs["ir_cutoff"]

    @property
    def omega_max(self) -> float:
        return self.cutoffs["uv_cutoff"]

    def _normalize_positions(self, value: Any) -> np.ndarray:
        positions = np.asarray(value, dtype=float)

        if positions.ndim == 0:
            positions = positions.reshape(1, 1)
        elif self.dimension == 1 and positions.ndim == 1:
            positions = positions.reshape(-1, 1)
        elif self.dimension == 2 and positions.ndim == 1:
            positions = positions.reshape(1, 2)

        if positions.ndim != 2 or positions.shape[1] != self.dimension:
            raise ValueError("atom_positions must have shape (n_atoms, dimension).")

        return positions

    def _normalize_cutoffs(self, cutoffs: dict[str, float]) -> dict[str, float]:
        if "ir_cutoff" not in cutoffs and "omega_min" in cutoffs:
            cutoffs["ir_cutoff"] = cutoffs["omega_min"]
        if "uv_cutoff" not in cutoffs and "omega_max" in cutoffs:
            cutoffs["uv_cutoff"] = cutoffs["omega_max"]

        if "ir_cutoff" not in cutoffs or "uv_cutoff" not in cutoffs:
            raise ValueError("cutoffs must define ir_cutoff and uv_cutoff.")

        cutoffs = {
            "ir_cutoff": float(cutoffs["ir_cutoff"]),
            "uv_cutoff": float(cutoffs["uv_cutoff"]),
        }
        if cutoffs["ir_cutoff"] < 0:
            raise ValueError("ir_cutoff must be non-negative for omega_p=||p||.")
        if cutoffs["uv_cutoff"] <= cutoffs["ir_cutoff"]:
            raise ValueError("uv_cutoff must be larger than ir_cutoff.")
        return cutoffs

    def _validate_photon(self) -> None:
        if "amplitudes" in self.photon:
            return

        if "momentum" not in self.photon and "p0" not in self.photon:
            if "omega_p" not in self.photon:
                raise ValueError("photon must define omega_p, momentum, p0 or amplitudes.")

        if "delta_k" not in self.photon:
            raise ValueError("Gaussian photon states must define delta_k.")

        if float(self.photon["delta_k"]) <= 0:
            raise ValueError("photon['delta_k'] must be positive.")

    def _validate_time(self) -> None:
        if "T" not in self.time or "dt" not in self.time:
            raise ValueError("time must define T and dt.")
        if float(self.time["T"]) <= 0 or float(self.time["dt"]) <= 0:
            raise ValueError("time['T'] and time['dt'] must be positive.")
