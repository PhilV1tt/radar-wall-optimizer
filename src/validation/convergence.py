"""
Études de convergence numérique pour la simulation FDTD.

Toutes les études utilisent un mur plat (params = zéros) et mesurent
l'énergie rétrodiffusée (∑Ez²) comme indicateur de convergence. Cette
métrique est cohérente avec la fonction fitness du GA.

Fonctions disponibles
---------------------
grid_convergence_study   — varie la résolution spatiale (ppw)
pml_convergence_study    — varie l'épaisseur des PML
time_convergence_study   — varie le nombre de pas de temps
"""

import numpy as np
from typing import List, Tuple

from src.fdtd.config import FDTDConfig

# Position du mur pour les études de convergence (centrée sur une grille 80x80)
_CONV_WALL_CX = 50
_CONV_WALL_CY = 40


def _run_flat_wall_energy(cfg: FDTDConfig, wall_height: int, wall_thickness: int) -> float:
    """Exécute une simulation FDTD mur plat et retourne l'énergie rétrodiffusée."""
    from src.fdtd.core import FDTD2D_TMz

    sim = FDTD2D_TMz(cfg)
    params = np.zeros(10)
    sim.set_wall_from_params(params, n_segments=10,
                              wall_height=wall_height, wall_thickness=wall_thickness)
    sim.run()
    return float(sim.compute_backscatter_energy())


def grid_convergence_study(
    ppw_list: List[int],
    base_nx: int = 80,
    base_ny: int = 80,
    base_n_steps: int = 350,
    freq: float = 10e9,
    n_pml: int = 12,
    wall_height: int = 25,
    wall_thickness: int = 4,
) -> Tuple[List[int], List[float]]:
    """Étude de convergence en résolution spatiale.

    Pour chaque valeur de ppw (points par longueur d'onde), lance une
    simulation FDTD avec mur plat et mesure l'énergie rétrodiffusée.
    La convergence est visible par la stabilisation de la valeur.

    Paramètres
    ----------
    ppw_list : List[int]
        Valeurs de résolution à tester, ex. [8, 10, 12, 15, 20].
    base_nx, base_ny : int
        Taille de la grille physique (en cellules, hors PML).
    base_n_steps : int
        Nombre de pas de temps.
    freq : float
        Fréquence (Hz).
    n_pml : int
        Épaisseur PML (cellules).
    wall_height : int
        Hauteur du mur en cellules.
    wall_thickness : int
        Épaisseur du mur en cellules.

    Retourne
    --------
    ppw_list : List[int]
        Résolutions testées.
    energy_values : List[float]
        Énergie rétrodiffusée pour chaque ppw.
    """
    energy_values = []

    for ppw in ppw_list:
        cfg = FDTDConfig(
            nx=base_nx, ny=base_ny,
            ppw=ppw, freq=freq,
            n_pml=n_pml, n_steps=base_n_steps,
            wall_center_x=_CONV_WALL_CX, wall_center_y=_CONV_WALL_CY,
        )
        energy = _run_flat_wall_energy(cfg, wall_height=wall_height, wall_thickness=wall_thickness)
        energy_values.append(energy)

    return ppw_list, energy_values


def pml_convergence_study(
    n_pml_list: List[int],
    ppw: int = 15,
    base_nx: int = 80,
    base_ny: int = 80,
    base_n_steps: int = 350,
    freq: float = 10e9,
    wall_height: int = 25,
    wall_thickness: int = 4,
) -> Tuple[List[int], List[float]]:
    """Étude de convergence en épaisseur de PML.

    Paramètres
    ----------
    n_pml_list : List[int]
        Épaisseurs PML à tester en cellules, ex. [4, 6, 8, 10, 12, 16].
    ppw : int
        Résolution spatiale (points par longueur d'onde).
    base_nx, base_ny : int
        Taille de la grille physique.
    base_n_steps : int
        Nombre de pas de temps.

    Retourne
    --------
    n_pml_list : List[int]
        Épaisseurs testées.
    energy_values : List[float]
        Énergie rétrodiffusée pour chaque épaisseur.
    """
    energy_values = []

    for n_pml in n_pml_list:
        cfg = FDTDConfig(
            nx=base_nx, ny=base_ny,
            ppw=ppw, freq=freq,
            n_pml=n_pml, n_steps=base_n_steps,
            wall_center_x=_CONV_WALL_CX, wall_center_y=_CONV_WALL_CY,
        )
        energy = _run_flat_wall_energy(cfg, wall_height=wall_height, wall_thickness=wall_thickness)
        energy_values.append(energy)

    return n_pml_list, energy_values


def time_convergence_study(
    n_steps_list: List[int],
    ppw: int = 15,
    base_nx: int = 80,
    base_ny: int = 80,
    freq: float = 10e9,
    n_pml: int = 12,
    wall_height: int = 25,
    wall_thickness: int = 4,
) -> Tuple[List[int], List[float]]:
    """Étude de convergence en durée de simulation.

    Paramètres
    ----------
    n_steps_list : List[int]
        Nombres de pas de temps à tester, ex. [200, 300, 400, 500, 600].

    Retourne
    --------
    n_steps_list : List[int]
        Durées testées.
    energy_values : List[float]
        Énergie rétrodiffusée pour chaque durée.
    """
    energy_values = []

    for n_steps in n_steps_list:
        cfg = FDTDConfig(
            nx=base_nx, ny=base_ny,
            ppw=ppw, freq=freq,
            n_pml=n_pml, n_steps=n_steps,
            wall_center_x=_CONV_WALL_CX, wall_center_y=_CONV_WALL_CY,
        )
        energy = _run_flat_wall_energy(cfg, wall_height=wall_height, wall_thickness=wall_thickness)
        energy_values.append(energy)

    return n_steps_list, energy_values
