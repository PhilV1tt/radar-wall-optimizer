"""
Initialisation des couches CPML (Convolutional Perfectly Matched Layer).
"""

import numpy as np
from src.fdtd.config import EPS0, ETA0


def _compute_cpml_coeffs(sigma, kappa, alpha, dt):
    """Calcule les coefficients b et a de la CPML."""
    b = np.exp(-(sigma / kappa + alpha) * dt / EPS0)
    a = np.zeros_like(sigma)
    denom = sigma * kappa + kappa ** 2 * alpha
    mask = denom > 1e-30
    a[mask] = sigma[mask] / denom[mask] * (b[mask] - 1.0)
    return b, a


def init_cpml(cfg):
    """Initialise les couches CPML sur les 4 bords.

    Retourne un dict avec tous les coefficients et champs auxiliaires Ψ.
    """
    n_pml = cfg.n_pml
    dx = cfg.dx
    dt = cfg.dt

    # Paramètres PML (valeurs stables et bien testées)
    m = 3           # ordre polynomial
    m_alpha = 1     # ordre pour α
    kappa_max = 1.0   # pas d'étirement κ (simplifié pour stabilité)
    alpha_max = 0.0   # pas de terme α (simplifié)

    # σ_max optimal (formule de Taflove, Sec. 7.7)
    sigma_max = (0.8 * (m + 1)) / (ETA0 * dx)

    # --- Profils 1D pour la direction x ---
    sigma_ex = np.zeros(cfg.nx_total)
    kappa_ex = np.ones(cfg.nx_total)
    alpha_ex = np.zeros(cfg.nx_total)
    sigma_hx = np.zeros(cfg.nx_total - 1)
    kappa_hx = np.ones(cfg.nx_total - 1)
    alpha_hx = np.zeros(cfg.nx_total - 1)

    for i in range(n_pml):
        d_e = (n_pml - i) / n_pml
        d_h = (n_pml - i - 0.5) / n_pml

        sigma_ex[i] = sigma_max * d_e ** m
        kappa_ex[i] = 1.0 + (kappa_max - 1.0) * d_e ** m
        alpha_ex[i] = alpha_max * (1.0 - d_e) ** m_alpha

        if d_h > 0:
            sigma_hx[i] = sigma_max * d_h ** m
            kappa_hx[i] = 1.0 + (kappa_max - 1.0) * d_h ** m
            alpha_hx[i] = alpha_max * (1.0 - d_h) ** m_alpha

        # Bord droit (symétrique)
        ir_e = cfg.nx_total - 1 - i
        ir_h = cfg.nx_total - 2 - i

        sigma_ex[ir_e] = sigma_max * d_e ** m
        kappa_ex[ir_e] = 1.0 + (kappa_max - 1.0) * d_e ** m
        alpha_ex[ir_e] = alpha_max * (1.0 - d_e) ** m_alpha

        if ir_h >= 0 and d_h > 0:
            sigma_hx[ir_h] = sigma_max * d_h ** m
            kappa_hx[ir_h] = 1.0 + (kappa_max - 1.0) * d_h ** m
            alpha_hx[ir_h] = alpha_max * (1.0 - d_h) ** m_alpha

    # --- Profils 1D pour la direction y ---
    sigma_ey = np.zeros(cfg.ny_total)
    kappa_ey = np.ones(cfg.ny_total)
    alpha_ey = np.zeros(cfg.ny_total)
    sigma_hy = np.zeros(cfg.ny_total - 1)
    kappa_hy = np.ones(cfg.ny_total - 1)
    alpha_hy = np.zeros(cfg.ny_total - 1)

    for j in range(n_pml):
        d_e = (n_pml - j) / n_pml
        d_h = (n_pml - j - 0.5) / n_pml

        sigma_ey[j] = sigma_max * d_e ** m
        kappa_ey[j] = 1.0 + (kappa_max - 1.0) * d_e ** m
        alpha_ey[j] = alpha_max * (1.0 - d_e) ** m_alpha

        if d_h > 0:
            sigma_hy[j] = sigma_max * d_h ** m
            kappa_hy[j] = 1.0 + (kappa_max - 1.0) * d_h ** m
            alpha_hy[j] = alpha_max * (1.0 - d_h) ** m_alpha

        jr_e = cfg.ny_total - 1 - j
        jr_h = cfg.ny_total - 2 - j

        sigma_ey[jr_e] = sigma_max * d_e ** m
        kappa_ey[jr_e] = 1.0 + (kappa_max - 1.0) * d_e ** m
        alpha_ey[jr_e] = alpha_max * (1.0 - d_e) ** m_alpha

        if jr_h >= 0 and d_h > 0:
            sigma_hy[jr_h] = sigma_max * d_h ** m
            kappa_hy[jr_h] = 1.0 + (kappa_max - 1.0) * d_h ** m
            alpha_hy[jr_h] = alpha_max * (1.0 - d_h) ** m_alpha

    # --- Coefficients CPML : b et a ---
    bx_e, ax_e = _compute_cpml_coeffs(sigma_ex, kappa_ex, alpha_ex, dt)
    bx_h, ax_h = _compute_cpml_coeffs(sigma_hx, kappa_hx, alpha_hx, dt)
    by_e, ay_e = _compute_cpml_coeffs(sigma_ey, kappa_ey, alpha_ey, dt)
    by_h, ay_h = _compute_cpml_coeffs(sigma_hy, kappa_hy, alpha_hy, dt)

    # Inverses de κ
    inv_kappa_ex = 1.0 / kappa_ex
    inv_kappa_hx = 1.0 / kappa_hx
    inv_kappa_ey = 1.0 / kappa_ey
    inv_kappa_hy = 1.0 / kappa_hy

    # --- Champs auxiliaires Ψ (convolutions CPML) ---
    nxt = cfg.nx_total
    nyt = cfg.ny_total

    return {
        'bx_e': bx_e, 'ax_e': ax_e,
        'bx_h': bx_h, 'ax_h': ax_h,
        'by_e': by_e, 'ay_e': ay_e,
        'by_h': by_h, 'ay_h': ay_h,
        'inv_kappa_ex': inv_kappa_ex,
        'inv_kappa_hx': inv_kappa_hx,
        'inv_kappa_ey': inv_kappa_ey,
        'inv_kappa_hy': inv_kappa_hy,
        'psi_ezx': np.zeros((nxt, nyt), dtype=np.float64),
        'psi_ezy': np.zeros((nxt, nyt), dtype=np.float64),
        'psi_hxy': np.zeros((nxt, nyt - 1), dtype=np.float64),
        'psi_hyx': np.zeros((nxt - 1, nyt), dtype=np.float64),
    }
