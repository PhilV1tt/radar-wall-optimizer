"""
Fonctions de fitness — wrapper FDTD pour l'optimisation.
"""

import numpy as np
from src.fdtd import FDTD2D_TMz, FDTDConfig

# Compteur global de simulations
sim_counter = 0


def evaluate_wall(params: np.ndarray, fdtd_config: FDTDConfig,
                  n_segments: int = 16, wall_height: int = 50,
                  wall_thickness: int = 4,
                  incidence_angles: list = None) -> float:
    """Évalue la RCS d'un profil de mur via simulation FDTD.

    Si incidence_angles contient plusieurs angles, la fitness est la
    somme des énergies rétrodiffusées pour chaque angle (robustesse angulaire).
    """
    global sim_counter

    if incidence_angles is None:
        incidence_angles = [0.0]

    total_energy = 0.0
    for angle in incidence_angles:
        sim_counter += 1
        cfg = FDTDConfig(
            nx=fdtd_config.nx, ny=fdtd_config.ny, ppw=fdtd_config.ppw,
            freq=fdtd_config.freq, courant=fdtd_config.courant,
            n_steps=fdtd_config.n_steps, n_pml=fdtd_config.n_pml,
            tfsf_margin=fdtd_config.tfsf_margin,
            wall_center_x=fdtd_config.wall_center_x,
            wall_center_y=fdtd_config.wall_center_y,
            incidence_angle=angle,
        )
        sim = FDTD2D_TMz(cfg)
        sim.set_wall_from_params(params, n_segments=n_segments,
                                  wall_height=wall_height,
                                  wall_thickness=wall_thickness)
        sim.run()
        total_energy += sim.compute_backscatter_energy()

    return total_energy
