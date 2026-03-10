"""
Classe FDTD2D_TMz allégée — importe depuis config, pml, tfsf, materials, ntff.
"""

import numpy as np
from typing import Optional, Tuple

from src.fdtd.config import FDTDConfig, RickerSource, MU0, EPS0
from src.fdtd.pml import init_cpml
from src.fdtd.tfsf import init_tfsf, update_tfsf
from src.fdtd.materials import (
    set_wall_geometry as _set_wall_geometry,
    set_wall_from_params as _set_wall_from_params,
    set_material as _set_material,
    set_wall_from_params_ram as _set_wall_from_params_ram,
)
from src.fdtd.ntff import (
    init_dft, update_dft,
    compute_ntff, get_incident_spectrum,
    compute_rcs_backscatter as _compute_rcs_backscatter,
    compute_backscatter_energy as _compute_backscatter_energy,
    compute_bistatic_rcs as _compute_bistatic_rcs,
)


class FDTD2D_TMz:
    """Simulateur FDTD 2D en polarisation TMz avec CPML.

    Polarisation TMz : Ez, Hx, Hy sont les composantes non-nulles.

    La grille de Yee en 2D TMz :
        Ez(i,j)       est défini aux nœuds entiers (i, j)
        Hx(i,j+1/2)  est défini à mi-chemin en y
        Hy(i+1/2,j)  est défini à mi-chemin en x

    La grille totale inclut les couches PML sur les 4 bords.
    """

    def __init__(self, config: FDTDConfig):
        self.cfg = config
        nxt = config.nx_total
        nyt = config.ny_total

        # --- Allocation des champs ---
        self.Ez = np.zeros((nxt, nyt), dtype=np.float64)
        self.Hx = np.zeros((nxt, nyt - 1), dtype=np.float64)
        self.Hy = np.zeros((nxt - 1, nyt), dtype=np.float64)

        # --- Coefficients de mise à jour (vide) ---
        self.Chxe = config.dt / (MU0 * config.dx)
        self.Chye = config.dt / (MU0 * config.dx)
        self.Cezh = config.dt / (EPS0 * config.dx)

        # --- Coefficients de matériau par cellule (pour RAM lossy) ---
        self.Ca = np.ones((nxt, nyt), dtype=np.float64)
        self.Cb = np.full((nxt, nyt), self.Cezh, dtype=np.float64)

        # Masque PEC (dans la grille totale)
        self.pec_mask = np.zeros((nxt, nyt), dtype=bool)

        # --- Source ---
        self.source = RickerSource(config.freq, config.dt, config.ppw)

        # --- CPML ---
        self._cpml = init_cpml(config)
        # Expose PML coefficients as attributes for backward compat / tests
        self.bx_e = self._cpml['bx_e']
        self.ax_e = self._cpml['ax_e']
        self.bx_h = self._cpml['bx_h']
        self.ax_h = self._cpml['ax_h']
        self.by_e = self._cpml['by_e']
        self.ay_e = self._cpml['ay_e']
        self.by_h = self._cpml['by_h']
        self.ay_h = self._cpml['ay_h']
        self.inv_kappa_ex = self._cpml['inv_kappa_ex']
        self.inv_kappa_hx = self._cpml['inv_kappa_hx']
        self.inv_kappa_ey = self._cpml['inv_kappa_ey']
        self.inv_kappa_hy = self._cpml['inv_kappa_hy']
        self.psi_ezx = self._cpml['psi_ezx']
        self.psi_ezy = self._cpml['psi_ezy']
        self.psi_hxy = self._cpml['psi_hxy']
        self.psi_hyx = self._cpml['psi_hyx']

        # --- TFSF ---
        self._tfsf = init_tfsf(config)
        self.tfsf_x0 = self._tfsf['tfsf_x0']
        self.tfsf_x1 = self._tfsf['tfsf_x1']
        self.tfsf_y0 = self._tfsf['tfsf_y0']
        self.tfsf_y1 = self._tfsf['tfsf_y1']
        self.theta_inc = self._tfsf['theta_inc']
        self.aux_size = self._tfsf['aux_size']
        self.ez1d = self._tfsf['ez1d']
        self.hy1d = self._tfsf['hy1d']
        self.aux_source_pos = self._tfsf['aux_source_pos']
        self.c_ez1d = self._tfsf['c_ez1d']
        self.c_hy1d = self._tfsf['c_hy1d']

        # --- DFT ---
        self._dft = init_dft(config)
        self.omega = self._dft['omega']
        self.dft_x0 = self._dft['dft_x0']
        self.dft_x1 = self._dft['dft_x1']
        self.dft_y0 = self._dft['dft_y0']
        self.dft_y1 = self._dft['dft_y1']
        self.n_dft_x = self._dft['n_dft_x']
        self.n_dft_y = self._dft['n_dft_y']
        # Expose DFT accumulators
        self.dft_ez_left = self._dft['dft_ez_left']
        self.dft_ez_right = self._dft['dft_ez_right']
        self.dft_ez_bottom = self._dft['dft_ez_bottom']
        self.dft_ez_top = self._dft['dft_ez_top']
        self.dft_hx_bottom = self._dft['dft_hx_bottom']
        self.dft_hx_top = self._dft['dft_hx_top']
        self.dft_hy_left = self._dft['dft_hy_left']
        self.dft_hy_right = self._dft['dft_hy_right']
        self.dft_ez_inc_left = self._dft['dft_ez_inc_left']
        self.dft_ez_inc_right = self._dft['dft_ez_inc_right']
        self.dft_ez_inc_bottom = self._dft['dft_ez_inc_bottom']
        self.dft_ez_inc_top = self._dft['dft_ez_inc_top']
        self.dft_hy_inc_left = self._dft['dft_hy_inc_left']
        self.dft_hy_inc_right = self._dft['dft_hy_inc_right']

        # Compteur de pas
        self.time_step = 0

    # ==========================================================================
    # Coordonnées : conversion grille physique ↔ grille totale
    # ==========================================================================
    def _phys_to_total(self, i: int, j: int) -> Tuple[int, int]:
        """Convertit des coordonnées physiques en coordonnées grille totale."""
        return i + self.cfg.n_pml, j + self.cfg.n_pml

    # ==========================================================================
    # Géométrie du mur (délégation)
    # ==========================================================================
    def set_wall_geometry(self, profile: np.ndarray):
        """Définit la géométrie du mur à partir d'un profil de surface."""
        _set_wall_geometry(self.pec_mask, profile, self.cfg)

    def set_wall_from_params(self, params: np.ndarray, n_segments: int = 20,
                              wall_height: int = 60, wall_thickness: int = 5):
        """Crée un profil de mur à partir de paramètres d'optimisation."""
        _set_wall_from_params(self.pec_mask, params, self.cfg,
                              n_segments, wall_height, wall_thickness)

    def set_material(self, region_mask: np.ndarray, sigma: float,
                     eps_r: float = 1.0):
        """Définit un matériau lossy (RAM) dans une région de la grille."""
        _set_material(self.Ca, self.Cb, region_mask, sigma, self.cfg, eps_r)

    def set_wall_from_params_ram(self, params: np.ndarray, n_segments: int = 20,
                                  wall_height: int = 60, wall_thickness: int = 5,
                                  ram_thickness: int = 2, ram_sigma: float = 0.5):
        """Crée un mur PEC avec une couche RAM."""
        _set_wall_from_params_ram(
            self.pec_mask, self.Ca, self.Cb, self.Cezh, params, self.cfg,
            n_segments, wall_height, wall_thickness, ram_thickness, ram_sigma)

    # ==========================================================================
    # Boucle de simulation principale
    # ==========================================================================
    def reset(self):
        """Remet tous les champs à zéro pour une nouvelle simulation."""
        self.Ez[:] = 0.0
        self.Hx[:] = 0.0
        self.Hy[:] = 0.0
        self.ez1d[:] = 0.0
        self.hy1d[:] = 0.0
        self.time_step = 0

        # Reset matériaux
        self.Ca[:] = 1.0
        self.Cb[:] = self.Cezh

        # Reset PML auxiliaires
        self.psi_ezx[:] = 0.0
        self.psi_ezy[:] = 0.0
        self.psi_hxy[:] = 0.0
        self.psi_hyx[:] = 0.0

        # Reset DFT (total)
        self.dft_ez_left[:] = 0
        self.dft_ez_right[:] = 0
        self.dft_ez_bottom[:] = 0
        self.dft_ez_top[:] = 0
        self.dft_hx_bottom[:] = 0
        self.dft_hx_top[:] = 0
        self.dft_hy_left[:] = 0
        self.dft_hy_right[:] = 0

        # Reset DFT (incident)
        self.dft_ez_inc_left[:] = 0
        self.dft_ez_inc_right[:] = 0
        self.dft_ez_inc_bottom[:] = 0
        self.dft_ez_inc_top[:] = 0
        self.dft_hy_inc_left[:] = 0
        self.dft_hy_inc_right[:] = 0

    def step(self):
        """Avance la simulation d'un pas de temps avec CPML.

        Ordre des opérations :
        1. Mise à jour de Hx et Hy avec CPML
        2. Corrections TFSF sur H
        3. Mise à jour de Ez avec CPML
        4. Corrections TFSF sur E
        5. Application du masque PEC
        6. Accumulation DFT
        """
        cfg = self.cfg
        nxt = cfg.nx_total
        nyt = cfg.ny_total

        # ==== 1. Mise à jour des champs magnétiques avec CPML ====
        dEz_dy = self.Ez[:, 1:] - self.Ez[:, :-1]
        self.psi_hxy[:, :] = (self.by_h[np.newaxis, :] * self.psi_hxy
                              + self.ay_h[np.newaxis, :] * dEz_dy)
        self.Hx -= self.Chxe * (
            dEz_dy * self.inv_kappa_hy[np.newaxis, :] + self.psi_hxy
        )

        dEz_dx = self.Ez[1:, :] - self.Ez[:-1, :]
        self.psi_hyx[:, :] = (self.bx_h[:, np.newaxis] * self.psi_hyx
                              + self.ax_h[:, np.newaxis] * dEz_dx)
        self.Hy += self.Chye * (
            dEz_dx * self.inv_kappa_hx[:, np.newaxis] + self.psi_hyx
        )

        # ==== 2-4. TFSF ====
        update_tfsf(self.Ez, self.Hx, self.Hy, self.Chxe, self.Chye,
                     self.Cezh, self._tfsf, self.source, self.time_step,
                     self.cfg)

        # ==== 3. Mise à jour du champ électrique avec CPML ====
        dHy_dx = np.zeros((nxt, nyt), dtype=np.float64)
        dHy_dx[1:-1, :] = self.Hy[1:, :] - self.Hy[:-1, :]

        dHx_dy = np.zeros((nxt, nyt), dtype=np.float64)
        dHx_dy[:, 1:-1] = self.Hx[:, 1:] - self.Hx[:, :-1]

        self.psi_ezx[:, :] = (self.bx_e[:, np.newaxis] * self.psi_ezx
                              + self.ax_e[:, np.newaxis] * dHy_dx)
        self.psi_ezy[:, :] = (self.by_e[np.newaxis, :] * self.psi_ezy
                              + self.ay_e[np.newaxis, :] * dHx_dy)

        curl_H_pml = (dHy_dx * self.inv_kappa_ex[:, np.newaxis] + self.psi_ezx
                      - dHx_dy * self.inv_kappa_ey[np.newaxis, :] - self.psi_ezy)
        self.Ez = self.Ca * self.Ez + self.Cb * curl_H_pml

        # ==== 5. PEC ====
        self.Ez[self.pec_mask] = 0.0

        # ==== Bords durs ====
        self.Ez[0, :] = 0.0
        self.Ez[-1, :] = 0.0
        self.Ez[:, 0] = 0.0
        self.Ez[:, -1] = 0.0

        # ==== 6. DFT running ====
        update_dft(self._dft, self.Ez, self.Hx, self.Hy,
                   self.ez1d, self.hy1d, self.time_step, self.cfg)

        self.time_step += 1

    def run(self, n_steps: Optional[int] = None) -> None:
        """Lance la simulation pour n_steps pas de temps."""
        if n_steps is None:
            n_steps = self.cfg.n_steps
        for _ in range(n_steps):
            self.step()

    # ==========================================================================
    # Calcul de la RCS
    # ==========================================================================
    def _compute_ntff(self, phi: float) -> complex:
        """Calcule le champ lointain DIFFUSÉ Ez^s dans la direction φ."""
        return compute_ntff(self._dft, phi, self.cfg)

    def compute_rcs_backscatter(self) -> float:
        """Calcule la RCS monostatique (rétrodiffusion, φ=π) via NTFF."""
        return _compute_rcs_backscatter(
            self._dft, self.source, self.time_step, self.cfg)

    def _get_incident_spectrum(self) -> complex:
        """Calcule le spectre du champ incident à la fréquence d'analyse."""
        return get_incident_spectrum(
            self.source, self.time_step, self.omega, self.cfg.dt)

    def compute_backscatter_energy(self) -> float:
        """Méthode simplifiée : énergie du champ diffusé dans la zone SF."""
        return _compute_backscatter_energy(self.Ez, self.cfg)

    def get_physical_fields(self) -> Tuple[np.ndarray, np.ndarray]:
        """Retourne Ez et pec_mask dans la grille physique (sans PML)."""
        n = self.cfg.n_pml
        ez_phys = self.Ez[n:n+self.cfg.nx, n:n+self.cfg.ny].copy()
        pec_phys = self.pec_mask[n:n+self.cfg.nx, n:n+self.cfg.ny].copy()
        return ez_phys, pec_phys

    def compute_bistatic_rcs(self, n_angles: int = 360) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule la RCS bistatique pour n_angles directions."""
        return _compute_bistatic_rcs(
            self._dft, self.source, self.time_step, self.cfg, n_angles)
