"""
DFT running, NTFF (Near-to-Far-Field Transform) et calcul de RCS.
"""

import numpy as _numpy
from src.utils.xp import xp as np, to_numpy
from typing import Tuple
from src.fdtd.config import C0, MU0


def init_dft(cfg):
    """Initialise les accumulateurs DFT pour la transformation NTFF.

    Retourne un dict avec les accumulateurs et les limites de la surface DFT.
    """
    n_pml = cfg.n_pml
    m = cfg.tfsf_margin

    dft_x0 = n_pml + m + 2
    dft_x1 = n_pml + cfg.nx - m - 3
    dft_y0 = n_pml + m + 2
    dft_y1 = n_pml + cfg.ny - m - 3

    n_dft_x = dft_x1 - dft_x0 + 1
    n_dft_y = dft_y1 - dft_y0 + 1

    omega = 2.0 * np.pi * cfg.freq

    return {
        'dft_x0': dft_x0, 'dft_x1': dft_x1,
        'dft_y0': dft_y0, 'dft_y1': dft_y1,
        'n_dft_x': n_dft_x, 'n_dft_y': n_dft_y,
        'omega': omega,
        # Champs totaux
        'dft_ez_left': np.zeros(n_dft_y, dtype=complex),
        'dft_ez_right': np.zeros(n_dft_y, dtype=complex),
        'dft_ez_bottom': np.zeros(n_dft_x, dtype=complex),
        'dft_ez_top': np.zeros(n_dft_x, dtype=complex),
        'dft_hx_bottom': np.zeros(n_dft_x, dtype=complex),
        'dft_hx_top': np.zeros(n_dft_x, dtype=complex),
        'dft_hy_left': np.zeros(n_dft_y, dtype=complex),
        'dft_hy_right': np.zeros(n_dft_y, dtype=complex),
        # Champ incident
        'dft_ez_inc_left': np.zeros(n_dft_y, dtype=complex),
        'dft_ez_inc_right': np.zeros(n_dft_y, dtype=complex),
        'dft_ez_inc_bottom': np.zeros(n_dft_x, dtype=complex),
        'dft_ez_inc_top': np.zeros(n_dft_x, dtype=complex),
        'dft_hy_inc_left': np.zeros(n_dft_y, dtype=complex),
        'dft_hy_inc_right': np.zeros(n_dft_y, dtype=complex),
    }


def update_dft(dft_state, Ez, Hx, Hy, ez1d, hy1d, time_step, cfg):
    """Accumule la DFT running des champs totaux ET incidents."""
    t_e = time_step * cfg.dt
    t_h = (time_step + 0.5) * cfg.dt
    omega = dft_state['omega']

    phase_e = np.exp(-1j * omega * t_e) * cfg.dt
    phase_h = np.exp(-1j * omega * t_h) * cfg.dt

    x0, x1 = dft_state['dft_x0'], dft_state['dft_x1']
    y0, y1 = dft_state['dft_y0'], dft_state['dft_y1']

    # === Champs totaux ===
    dft_state['dft_ez_left'] += Ez[x0, y0:y1+1] * phase_e
    dft_state['dft_hy_left'] += Hy[x0-1, y0:y1+1] * phase_h

    dft_state['dft_ez_right'] += Ez[x1, y0:y1+1] * phase_e
    dft_state['dft_hy_right'] += Hy[x1, y0:y1+1] * phase_h

    dft_state['dft_ez_bottom'] += Ez[x0:x1+1, y0] * phase_e
    dft_state['dft_hx_bottom'] += Hx[x0:x1+1, y0-1] * phase_h

    dft_state['dft_ez_top'] += Ez[x0:x1+1, y1] * phase_e
    dft_state['dft_hx_top'] += Hx[x0:x1+1, y1] * phase_h

    # === Champ incident (grille 1D) ===
    dft_state['dft_ez_inc_left'] += ez1d[x0] * phase_e
    dft_state['dft_hy_inc_left'] += hy1d[x0 - 1] * phase_h

    dft_state['dft_ez_inc_right'] += ez1d[x1] * phase_e
    dft_state['dft_hy_inc_right'] += hy1d[x1] * phase_h

    i_indices = np.arange(x0, x1 + 1)
    dft_state['dft_ez_inc_bottom'] += ez1d[i_indices] * phase_e
    dft_state['dft_ez_inc_top'] += ez1d[i_indices] * phase_e


def compute_ntff(dft_state, phi, cfg) -> complex:
    """Calcule le champ lointain DIFFUSÉ Ez^s dans la direction φ via NTFF."""
    k = 2.0 * np.pi * cfg.freq / C0
    dx = cfg.dx

    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    x0, x1 = dft_state['dft_x0'], dft_state['dft_x1']
    y0, y1 = dft_state['dft_y0'], dft_state['dft_y1']

    N_z = 0.0 + 0.0j
    L_x = 0.0 + 0.0j
    L_y = 0.0 + 0.0j

    # Champs diffusés = totaux - incidents
    scat_ez_left = dft_state['dft_ez_left'] - dft_state['dft_ez_inc_left']
    scat_hy_left = dft_state['dft_hy_left'] - dft_state['dft_hy_inc_left']
    scat_ez_right = dft_state['dft_ez_right'] - dft_state['dft_ez_inc_right']
    scat_hy_right = dft_state['dft_hy_right'] - dft_state['dft_hy_inc_right']
    scat_ez_bottom = dft_state['dft_ez_bottom'] - dft_state['dft_ez_inc_bottom']
    scat_hx_bottom = dft_state['dft_hx_bottom']
    scat_ez_top = dft_state['dft_ez_top'] - dft_state['dft_ez_inc_top']
    scat_hx_top = dft_state['dft_hx_top']

    # Bord gauche (n̂ = -x̂)
    j_arr = np.arange(y0, y1 + 1, dtype=np.float64)
    rpc = x0 * dx * cos_phi + j_arr * dx * sin_phi
    phase = np.exp(1j * k * rpc)
    N_z += np.sum(-scat_hy_left * phase) * dx
    L_y += np.sum(-scat_ez_left * phase) * dx

    # Bord droit (n̂ = +x̂)
    rpc = x1 * dx * cos_phi + j_arr * dx * sin_phi
    phase = np.exp(1j * k * rpc)
    N_z += np.sum(scat_hy_right * phase) * dx
    L_y += np.sum(scat_ez_right * phase) * dx

    # Bord bas (n̂ = -ŷ)
    i_arr = np.arange(x0, x1 + 1, dtype=np.float64)
    rpc = i_arr * dx * cos_phi + y0 * dx * sin_phi
    phase = np.exp(1j * k * rpc)
    N_z += np.sum(scat_hx_bottom * phase) * dx
    L_x += np.sum(scat_ez_bottom * phase) * dx

    # Bord haut (n̂ = +ŷ)
    rpc = i_arr * dx * cos_phi + y1 * dx * sin_phi
    phase = np.exp(1j * k * rpc)
    N_z += np.sum(-scat_hx_top * phase) * dx
    L_x += np.sum(-scat_ez_top * phase) * dx

    L_phi = -L_x * sin_phi + L_y * cos_phi
    omega = 2.0 * np.pi * cfg.freq
    Ez_far = omega * MU0 * N_z + k * L_phi
    return Ez_far


def get_incident_spectrum(source, time_step, omega, dt) -> complex:
    """Calcule le spectre du champ incident à la fréquence d'analyse."""
    n_arr = np.arange(time_step)
    source_vals = np.array([source(n) for n in range(time_step)])
    phases = np.exp(-1j * omega * n_arr * dt) * dt
    return np.sum(source_vals * phases)


def compute_rcs_backscatter(dft_state, source, time_step, cfg) -> float:
    """Calcule la RCS monostatique (rétrodiffusion, φ=π) via NTFF."""
    k = 2.0 * np.pi * cfg.freq / C0
    Ez_far = compute_ntff(dft_state, np.pi, cfg)

    omega = dft_state['omega']
    E_inc_spectrum = get_incident_spectrum(source, time_step, omega, cfg.dt)
    if abs(E_inc_spectrum) < 1e-30:
        return 1e6

    rcs = (k / (4.0 * np.pi)) * abs(Ez_far)**2 / abs(E_inc_spectrum)**2
    return float(rcs / cfg.wavelength)


def compute_backscatter_energy(Ez, cfg) -> float:
    """Méthode simplifiée : énergie du champ diffusé dans la zone SF."""
    n_pml = cfg.n_pml
    m = cfg.tfsf_margin

    measure_x_start = n_pml + m + 2
    measure_x_end = n_pml + cfg.wall_center_x - 20
    measure_y_start = n_pml + m + 2
    measure_y_end = n_pml + cfg.ny - m - 2

    measure_x = slice(measure_x_start, measure_x_end)
    measure_y = slice(measure_y_start, measure_y_end)

    scattered_energy = np.sum(Ez[measure_x, measure_y]**2)
    return float(scattered_energy)


def compute_bistatic_rcs(dft_state, source, time_step, cfg,
                          n_angles=360) -> Tuple[np.ndarray, np.ndarray]:
    """Calcule la RCS bistatique pour n_angles directions via NTFF."""
    k = 2.0 * np.pi * cfg.freq / C0
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    rcs = np.zeros(n_angles)

    omega = dft_state['omega']
    E_inc = get_incident_spectrum(source, time_step, omega, cfg.dt)
    if abs(E_inc) < 1e-30:
        return angles, np.ones(n_angles) * 1e6

    for a_idx, phi in enumerate(angles):
        Ez_far = compute_ntff(dft_state, phi, cfg)
        rcs[a_idx] = (k / (4*np.pi)) * abs(Ez_far)**2 / abs(E_inc)**2

    return to_numpy(angles), to_numpy(rcs / cfg.wavelength)
