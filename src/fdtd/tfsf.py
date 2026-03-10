"""
Fonctions TFSF (Total-Field / Scattered-Field).
"""

import numpy as np
from typing import Tuple
from src.fdtd.config import C0, ETA0


def init_tfsf(cfg):
    """Initialise la frontière TFSF avec grille 1D auxiliaire.

    Retourne un dict avec les limites TFSF et la grille 1D.
    """
    m = cfg.tfsf_margin
    n_pml = cfg.n_pml

    tfsf_x0 = n_pml + m
    tfsf_x1 = n_pml + cfg.nx - m - 1
    tfsf_y0 = n_pml + m
    tfsf_y1 = n_pml + cfg.ny - m - 1

    theta_inc = cfg.incidence_angle

    # Grille 1D auxiliaire (utilisée uniquement pour θ=0)
    aux_size = cfg.nx_total + 4 * m
    ez1d = np.zeros(aux_size)
    hy1d = np.zeros(aux_size)
    aux_source_pos = m

    # Coefficients 1D
    c_ez1d = cfg.courant * ETA0
    c_hy1d = cfg.courant / ETA0

    return {
        'tfsf_x0': tfsf_x0, 'tfsf_x1': tfsf_x1,
        'tfsf_y0': tfsf_y0, 'tfsf_y1': tfsf_y1,
        'theta_inc': theta_inc,
        'aux_size': aux_size,
        'ez1d': ez1d, 'hy1d': hy1d,
        'aux_source_pos': aux_source_pos,
        'c_ez1d': c_ez1d, 'c_hy1d': c_hy1d,
    }


def get_incident_field(theta: float, source, x: float, y: float,
                       t: float) -> Tuple[float, float, float]:
    """Calcule le champ incident analytique à la position (x,y) au temps t.

    Pour une onde plane TMz à angle θ :
        Ez_inc = f(t - (x·cosθ + y·sinθ)/c)
        Hx_inc = -(sinθ/η₀) × f(...)
        Hy_inc = (cosθ/η₀) × f(...)
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    t_delayed = t - (x * cos_t + y * sin_t) / C0
    t_src = t_delayed - source.t_delay
    arg = (np.pi * source.fp * t_src) ** 2
    f_val = (1.0 - 2.0 * arg) * np.exp(-arg)

    ez = f_val
    hx = -sin_t / ETA0 * f_val
    hy = cos_t / ETA0 * f_val
    return ez, hx, hy


def update_tfsf(Ez, Hx, Hy, Chxe, Chye, Cezh, tfsf_state, source,
                time_step, cfg):
    """Met à jour la grille 1D auxiliaire et applique les corrections TFSF."""
    x0 = tfsf_state['tfsf_x0']
    x1 = tfsf_state['tfsf_x1']
    y0 = tfsf_state['tfsf_y0']
    y1 = tfsf_state['tfsf_y1']
    theta_inc = tfsf_state['theta_inc']
    ez1d = tfsf_state['ez1d']
    hy1d = tfsf_state['hy1d']
    aux_source_pos = tfsf_state['aux_source_pos']
    c_ez1d = tfsf_state['c_ez1d']
    c_hy1d = tfsf_state['c_hy1d']

    if abs(theta_inc) < 1e-10:
        # === Incidence normale : grille 1D ===
        hy1d[-1] = hy1d[-2]
        hy1d[:-1] += c_hy1d * (ez1d[1:] - ez1d[:-1])
        hy1d[aux_source_pos] -= source(time_step) / ETA0
        ez1d[0] = ez1d[1]
        ez1d[1:] += c_ez1d * (hy1d[1:] - hy1d[:-1])
        ez1d[aux_source_pos + 1] += source(time_step)

        # Corrections 2D
        Hy[x0 - 1, y0:y1 + 1] -= Chye * ez1d[x0]
        Hy[x1, y0:y1 + 1] += Chye * ez1d[x1]
        Ez[x0, y0:y1 + 1] -= Cezh * hy1d[x0 - 1]
        Ez[x1, y0:y1 + 1] += Cezh * hy1d[x1]
    else:
        # === Incidence oblique : champs analytiques ===
        dx = cfg.dx
        dt = cfg.dt
        t_e = time_step * dt
        t_h = (time_step + 0.5) * dt

        # Bord gauche (x = x0)
        for j in range(y0, y1 + 1):
            x_pos = x0 * dx
            y_pos = j * dx
            ez_inc, _, hy_inc = get_incident_field(theta_inc, source,
                                                    x_pos, y_pos, t_h)
            Hy[x0 - 1, j] -= Chye * ez_inc
            ez_inc_e, _, hy_inc_e = get_incident_field(
                theta_inc, source, (x0 - 0.5) * dx, y_pos, t_e)
            Ez[x0, j] -= Cezh * hy_inc_e

        # Bord droit (x = x1)
        for j in range(y0, y1 + 1):
            x_pos = x1 * dx
            y_pos = j * dx
            ez_inc, _, hy_inc = get_incident_field(theta_inc, source,
                                                    x_pos, y_pos, t_h)
            Hy[x1, j] += Chye * ez_inc
            ez_inc_e, _, hy_inc_e = get_incident_field(
                theta_inc, source, (x1 + 0.5) * dx, y_pos, t_e)
            Ez[x1, j] += Cezh * hy_inc_e

        # Bord bas (y = y0)
        for i in range(x0, x1 + 1):
            x_pos = i * dx
            y_pos = y0 * dx
            ez_inc, hx_inc, _ = get_incident_field(theta_inc, source,
                                                    x_pos, y_pos, t_h)
            Hx[i, y0 - 1] += Chxe * ez_inc
            ez_inc_e, hx_inc_e, _ = get_incident_field(
                theta_inc, source, x_pos, (y0 - 0.5) * dx, t_e)
            Ez[i, y0] += Cezh * hx_inc_e

        # Bord haut (y = y1)
        for i in range(x0, x1 + 1):
            x_pos = i * dx
            y_pos = y1 * dx
            ez_inc, hx_inc, _ = get_incident_field(theta_inc, source,
                                                    x_pos, y_pos, t_h)
            Hx[i, y1] -= Chxe * ez_inc
            ez_inc_e, hx_inc_e, _ = get_incident_field(
                theta_inc, source, x_pos, (y1 + 0.5) * dx, t_e)
            Ez[i, y1] -= Cezh * hx_inc_e
