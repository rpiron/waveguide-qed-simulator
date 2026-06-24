import numpy as np
from matplotlib.patches import Circle, Rectangle

from src.xp_config import ExperimentConfig


def _atom_parameter_array(value, n_atoms: int, dtype, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        return np.full(n_atoms, array.item(), dtype=dtype)

    array = array.reshape(-1)
    if len(array) != n_atoms:
        raise ValueError(f"{name} must be scalar or contain one value per atom.")
    return array


def radius_label(radius: float) -> str:
    return f"{radius:g}".replace("-", "m").replace(".", "p")


def geometry_filename(n_atoms: int, radius: float) -> str:
    return f"matter_geom_NA{int(n_atoms)}_R{radius_label(radius)}.csv"


def slab_geometry_filename(n_atoms: int, n_layers: int) -> str:
    return f"matter_geom_NA{int(n_atoms)}_NL{int(n_layers)}.csv"


def regular_disk_positions(n_atoms: int, radius: float) -> np.ndarray:
    """
    Deterministic nearly-regular atom positions in a disk centered at the origin.
    """

    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive.")
    if radius <= 0:
        raise ValueError("radius must be positive.")
    if n_atoms == 1:
        return np.zeros((1, 2))

    positions = [np.zeros(2)]
    remaining = n_atoms - 1
    n_rings = max(1, int(np.ceil(np.sqrt(remaining / 6))))

    weights = np.arange(1, n_rings + 1, dtype=float)
    raw_counts = remaining * weights / np.sum(weights)
    ring_counts = np.floor(raw_counts).astype(int)

    missing = remaining - int(np.sum(ring_counts))
    if missing:
        order = np.argsort(raw_counts - ring_counts)[::-1]
        ring_counts[order[:missing]] += 1

    for ring_index, count in enumerate(ring_counts, start=1):
        ring_radius = radius * ring_index / n_rings
        angle_offset = np.pi / count if ring_index % 2 == 0 else 0.0
        angles = 2 * np.pi * np.arange(count) / count + angle_offset
        ring_positions = np.column_stack(
            (ring_radius * np.cos(angles), ring_radius * np.sin(angles))
        )
        positions.extend(ring_positions)

    return np.asarray(positions[:n_atoms], dtype=float)


def regular_slab_positions(
    n_atoms: int,
    n_layers: int,
    L2: float,
    slab_width: float | None = None,
) -> np.ndarray:
    """
    Deterministic vertical slab centered at the origin.

    Layers are parallel to the transverse direction y and are placed regularly
    along x. Atoms in each layer span the full fiber height.
    """

    if n_atoms <= 0:
        raise ValueError("n_atoms must be positive.")
    if n_layers <= 0:
        raise ValueError("n_layers must be positive.")
    if n_layers > n_atoms:
        raise ValueError("n_layers must be smaller than or equal to n_atoms.")
    if L2 <= 0:
        raise ValueError("L2 must be positive.")

    if slab_width is None:
        slab_width = L2 / 2 if n_layers > 1 else 0.0
    if slab_width < 0:
        raise ValueError("slab_width must be non-negative.")

    counts = np.full(n_layers, n_atoms // n_layers, dtype=int)
    remaining = n_atoms - int(np.sum(counts))
    if remaining:
        layer_centers = np.arange(n_layers) - (n_layers - 1) / 2
        central_order = np.argsort(np.abs(layer_centers), kind="mergesort")
        counts[central_order[:remaining]] += 1

    if n_layers == 1:
        x_values = np.array([0.0])
    else:
        x_values = np.linspace(-slab_width / 2, slab_width / 2, n_layers)

    positions = []
    for x_value, count in zip(x_values, counts):
        if count == 1:
            y_values = np.array([0.0])
        else:
            y_values = np.linspace(-L2 / 2, L2 / 2, count)
        positions.extend((x_value, y_value) for y_value in y_values)

    return np.asarray(positions, dtype=float)


def save_geometry_csv(
    path,
    L1: float,
    L2: float,
    interface_radius: float,
    atom_positions: np.ndarray,
    n_layers: int = 0,
) -> None:
    atom_positions = np.asarray(atom_positions, dtype=float)
    if atom_positions.ndim != 2 or atom_positions.shape[1] != 2:
        raise ValueError("atom_positions must have shape (n_atoms, 2).")

    path.parent.mkdir(parents=True, exist_ok=True)
    n_atoms = len(atom_positions)
    data = np.column_stack(
        (
            np.arange(n_atoms, dtype=int),
            atom_positions[:, 0],
            atom_positions[:, 1],
            L1 * np.ones(n_atoms),
            L2 * np.ones(n_atoms),
            interface_radius * np.ones(n_atoms),
            n_atoms * np.ones(n_atoms, dtype=int),
            n_layers * np.ones(n_atoms, dtype=int),
        )
    )
    header = "atom_index,x,y,L1,L2,interface_radius,N_A,N_layer"
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt="%g")


def load_geometry_csv(path) -> dict[str, np.ndarray | float | int]:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data])

    atom_positions = np.column_stack((data["x"], data["y"]))
    names = data.dtype.names
    interface_radius = float(data["interface_radius"][0])
    n_layers = int(data["N_layer"][0]) if "N_layer" in names else 0

    return {
        "atom_positions": atom_positions,
        "L1": float(data["L1"][0]),
        "L2": float(data["L2"][0]),
        "interface_radius": interface_radius,
        "N_A": int(data["N_A"][0]),
        "N_layer": n_layers,
    }


def make_fiber_config(
    L1: float,
    L2: float,
    atom_positions: np.ndarray,
    *,
    Omega_j,
    d_j,
    omega_init: float,
    delta_k: float,
    T: float,
    dt: float,
    cutoffs: dict[str, float],
) -> ExperimentConfig:
    n_atoms = len(atom_positions)
    Omega_j = _atom_parameter_array(Omega_j, n_atoms, float, "Omega_j")
    d_j = _atom_parameter_array(d_j, n_atoms, float, "d_j")

    return ExperimentConfig(
        dimension=2,
        lengths=[L1, L2],
        atom_positions=atom_positions,
        atom_frequencies=Omega_j,
        atom_couplings=d_j,
        photon={
            "momentum": [omega_init, 0.0],
            "delta_k": delta_k,
            "position": [-L1 / 4, 0.0],
            "direction": [1.0, 0.0],
        },
        time={"T": T, "dt": dt},
        cutoffs=cutoffs,
    )


def plot_fiber_geometry(
    ax,
    L1: float,
    L2: float,
    atom_positions: np.ndarray,
    interface_radius: float,
    photon_position: tuple[float, float] | list[float],
) -> None:
    fiber = Rectangle(
        (-L1 / 2, -L2 / 2),
        L1,
        L2,
        fill=False,
        edgecolor="#2f3a45",
        linewidth=1.2,
    )
    interface = Circle(
        (0.0, 0.0),
        interface_radius,
        fill=False,
        edgecolor="#25855a",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
    )

    ax.add_patch(fiber)
    ax.add_patch(interface)
    ax.scatter(
        atom_positions[:, 0],
        atom_positions[:, 1],
        s=34,
        color="#1f8f4d",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.scatter(
        [photon_position[0]],
        [photon_position[1]],
        marker=">",
        s=60,
        color="#3759a8",
        zorder=4,
    )
    ax.arrow(
        photon_position[0] + 0.4,
        photon_position[1],
        2.0,
        0.0,
        width=0.025,
        head_width=0.22,
        head_length=0.5,
        color="#3759a8",
        length_includes_head=True,
    )

    ax.set_xlim(-L1 / 2 - 1, L1 / 2 + 1)
    ax.set_ylim(-L2 / 2 - 1, L2 / 2 + 1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(color="0.9", linewidth=0.5)


def plot_slab_geometry(
    ax,
    L1: float,
    L2: float,
    atom_positions: np.ndarray,
    n_layers: int,
    photon_position: tuple[float, float] | list[float],
) -> None:
    x_margin = max(0.02 * L1, 0.5)
    y_margin = max(0.12 * L2, 0.02)
    arrow_width = max(0.02 * L2, 0.003)
    arrow_head_width = max(0.12 * L2, 0.015)
    arrow_head_length = max(0.01 * L1, 0.25)

    fiber = Rectangle(
        (-L1 / 2, -L2 / 2),
        L1,
        L2,
        fill=False,
        edgecolor="#2f3a45",
        linewidth=1.2,
    )
    ax.add_patch(fiber)

    x_values = np.unique(np.round(atom_positions[:, 0], decimals=12))
    if len(x_values) == 1:
        ax.axvline(x_values[0], color="#5e6874", linestyle="--", linewidth=1.0)
    else:
        slab_left = float(np.min(x_values))
        slab_right = float(np.max(x_values))
        slab = Rectangle(
            (slab_left, -L2 / 2),
            slab_right - slab_left,
            L2,
            fill=False,
            edgecolor="#5e6874",
            linestyle="--",
            linewidth=1.0,
            alpha=0.9,
        )
        ax.add_patch(slab)

    ax.scatter(
        atom_positions[:, 0],
        atom_positions[:, 1],
        s=34,
        color="#1f8f4d",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.scatter(
        [photon_position[0]],
        [photon_position[1]],
        marker=">",
        s=60,
        color="#3759a8",
        zorder=4,
    )
    ax.arrow(
        photon_position[0] + 0.4,
        photon_position[1],
        2.0,
        0.0,
        width=arrow_width,
        head_width=arrow_head_width,
        head_length=arrow_head_length,
        color="#3759a8",
        length_includes_head=True,
    )

    ax.set_xlim(-L1 / 2 - x_margin, L1 / 2 + x_margin)
    ax.set_ylim(-L2 / 2 - y_margin, L2 / 2 + y_margin)
    if L2 / L1 < 0.04:
        ax.set_aspect("auto")
    else:
        ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(color="0.9", linewidth=0.5)
