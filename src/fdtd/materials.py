"""
Fonctions de géométrie du mur et matériaux (PEC, RAM).
"""

from src.utils.xp import xp as np, to_numpy
from src.fdtd.config import EPS0


def set_wall_geometry(pec_mask, profile, cfg):
    """Définit la géométrie du mur à partir d'un profil de surface."""
    pec_mask[:] = False
    n_pml = cfg.n_pml
    cx = cfg.wall_center_x + n_pml

    for j in range(len(profile)):
        surface_x = cx + int(profile[j])
        jt = j + n_pml
        if 0 <= surface_x < cfg.nx_total and 0 <= jt < cfg.ny_total:
            pec_mask[surface_x:, jt] = True


def set_wall_from_params(pec_mask, params, cfg, n_segments=20,
                          wall_height=60, wall_thickness=5):
    """Crée un profil de mur à partir de paramètres d'optimisation."""
    params = np.asarray(params)  # ensure xp-compatible array (numpy→cupy if GPU)
    pec_mask[:] = False
    n_pml = cfg.n_pml
    cx = cfg.wall_center_x + n_pml
    cy = cfg.wall_center_y + n_pml

    half_h = wall_height // 2
    y_start = max(n_pml, cy - half_h)
    y_end = min(n_pml + cfg.ny, cy + half_h)

    y_control = np.linspace(0, 1, n_segments)
    y_wall = np.linspace(0, 1, y_end - y_start)

    max_displacement = 10
    displacements = np.interp(y_wall, y_control, params) * max_displacement

    displacements_cpu = to_numpy(displacements)
    for idx, j in enumerate(range(y_start, y_end)):
        surface_x = cx + int(displacements_cpu[idx])
        x_start = max(0, surface_x)
        x_end = min(cfg.nx_total, surface_x + wall_thickness)
        if x_start < x_end:
            pec_mask[x_start:x_end, j] = True


def set_material(Ca, Cb, region_mask, sigma, cfg, eps_r=1.0):
    """Définit un matériau lossy (RAM) dans une région de la grille."""
    dt = cfg.dt
    eps = eps_r * EPS0
    dx = cfg.dx

    denom = 1.0 + sigma * dt / (2.0 * eps)
    Ca[region_mask] = (1.0 - sigma * dt / (2.0 * eps)) / denom
    Cb[region_mask] = (dt / eps) / denom / dx


def set_wall_from_params_ram(pec_mask, Ca, Cb, Cezh, params, cfg,
                              n_segments=20, wall_height=60,
                              wall_thickness=5, ram_thickness=2,
                              ram_sigma=0.5):
    """Crée un mur PEC avec une couche RAM (matériau absorbant radar)."""
    n_pml = cfg.n_pml
    cx = cfg.wall_center_x + n_pml
    cy = cfg.wall_center_y + n_pml

    half_h = wall_height // 2
    y_start = max(n_pml, cy - half_h)
    y_end = min(n_pml + cfg.ny, cy + half_h)

    params = np.asarray(params)  # ensure xp-compatible array
    # Séparer profil et conductivité
    if len(params) > n_segments:
        geom_params = params[:n_segments]
        sigma_params = np.abs(params[n_segments:2*n_segments]) * 2.0
    else:
        geom_params = params
        sigma_params = np.full(n_segments, ram_sigma)

    # Reset
    pec_mask[:] = False
    Ca[:] = 1.0
    Cb[:] = Cezh

    y_control = np.linspace(0, 1, n_segments)
    y_wall = np.linspace(0, 1, y_end - y_start)

    max_displacement = 10
    displacements = np.interp(y_wall, y_control, geom_params) * max_displacement
    sigmas = np.interp(y_wall, y_control, sigma_params)

    dt = cfg.dt
    displacements_cpu = to_numpy(displacements)
    sigmas_cpu = to_numpy(sigmas)

    for idx, j in enumerate(range(y_start, y_end)):
        surface_x = cx + int(displacements_cpu[idx])

        # Couche RAM (devant le mur PEC)
        ram_start = max(0, surface_x - ram_thickness)
        ram_end = max(0, surface_x)
        if ram_start < ram_end:
            sigma_val = float(sigmas_cpu[idx])
            eps = EPS0
            denom = 1.0 + sigma_val * dt / (2.0 * eps)
            Ca[ram_start:ram_end, j] = (1.0 - sigma_val * dt / (2.0 * eps)) / denom
            Cb[ram_start:ram_end, j] = (dt / eps) / denom / cfg.dx

        # Mur PEC
        x_start = max(0, surface_x)
        x_end = min(cfg.nx_total, surface_x + wall_thickness)
        if x_start < x_end:
            pec_mask[x_start:x_end, j] = True
