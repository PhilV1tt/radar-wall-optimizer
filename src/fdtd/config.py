"""
Constantes physiques, configuration FDTD et source de Ricker.
"""

import numpy as np
from dataclasses import dataclass, field

# ==============================================================================
# Constantes physiques
# ==============================================================================
C0 = 299_792_458.0          # Vitesse de la lumière (m/s)
MU0 = 4.0 * np.pi * 1e-7   # Perméabilité du vide (H/m)
EPS0 = 1.0 / (MU0 * C0**2) # Permittivité du vide (F/m)
ETA0 = np.sqrt(MU0 / EPS0) # Impédance du vide ≈ 377 Ω


@dataclass
class FDTDConfig:
    """Configuration de la simulation FDTD 2D TMz.

    Paramètres
    ----------
    nx, ny : int
        Nombre de cellules selon x et y (hors PML).
    freq : float
        Fréquence centrale du radar (Hz).
    ppw : int
        Points par longueur d'onde (contrôle la résolution).
    n_pml : int
        Épaisseur de la couche PML en cellules.
    courant : float
        Nombre de Courant Sc = c·Δt/Δx. Stabilité requiert Sc ≤ 1/√2 en 2D.
    n_steps : int
        Nombre de pas de temps.
    wall_center_x : int
        Position x du centre du mur (en cellules, dans la grille physique).
    wall_center_y : int
        Position y du centre du mur (en cellules, dans la grille physique).
    """
    # Grille (taille physique, hors PML)
    nx: int = 200
    ny: int = 200
    ppw: int = 20        # points par longueur d'onde
    freq: float = 10e9   # 10 GHz (bande X, radar typique)

    # Paramètres dérivés (calculés dans __post_init__)
    dx: float = field(init=False)
    dt: float = field(init=False)
    wavelength: float = field(init=False)
    courant: float = 0.5  # Sc = c·dt/dx — bien en-dessous de 1/√2 ≈ 0.707
    n_steps: int = 500

    # PML
    n_pml: int = 12  # épaisseur PML en cellules

    # TFSF
    tfsf_margin: int = 15  # marge en cellules autour du TFSF (dans la grille physique)

    # Angle d'incidence (0 = normal, en radians)
    incidence_angle: float = 0.0

    # Mur (coordonnées dans la grille physique)
    wall_center_x: int = 120
    wall_center_y: int = 100

    # Taille totale incluant PML (calculée)
    nx_total: int = field(init=False)
    ny_total: int = field(init=False)

    def __post_init__(self):
        self.wavelength = C0 / self.freq
        self.dx = self.wavelength / self.ppw
        self.dt = self.courant * self.dx / C0
        self.nx_total = self.nx + 2 * self.n_pml
        self.ny_total = self.ny + 2 * self.n_pml


class RickerSource:
    """Source de Ricker (dérivée seconde de Gaussienne).

    La wavelet de Ricker est le signal standard en simulation FDTD car :
    1. Son spectre est borné (pas de composantes DC problématiques)
    2. Elle a un contenu fréquentiel bien défini centré sur fp
    3. La fréquence maximale est environ 2.5 × fp

    Forme temporelle : (1 - 2(πfp·t')²) exp(-(πfp·t')²)
    où t' = t - t_delay est le temps retardé.
    """

    def __init__(self, fp: float, dt: float, ppw: int):
        self.fp = fp
        self.dt = dt
        self.ppw = ppw
        # Retard pour que le pulse démarre à ~0 au temps t=0
        self.t_delay = 1.0 / fp

    def __call__(self, time_step: int) -> float:
        t = time_step * self.dt - self.t_delay
        arg = (np.pi * self.fp * t) ** 2
        return (1.0 - 2.0 * arg) * np.exp(-arg)

    def frequency_content(self, time_step: int) -> float:
        """Retourne le spectre instantané pour monitoring."""
        return self(time_step)
