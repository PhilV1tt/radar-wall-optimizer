"""
Animation FDTD en temps réel.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from src.fdtd import FDTD2D_TMz, FDTDConfig


def run_fdtd_animated(params, ax, fdtd_cfg, n_segments=16,
                       wall_height=50, wall_thickness=4,
                       title="", n_steps=300, frame_skip=5):
    """Lance une simulation FDTD et anime le champ Ez en temps réel."""
    sim = FDTD2D_TMz(fdtd_cfg)
    sim.set_wall_from_params(params, n_segments, wall_height, wall_thickness)

    vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(sim.Ez.T, origin='lower', cmap='RdBu_r', norm=norm,
                   aspect='equal', interpolation='bilinear')

    pec_display = np.ma.masked_where(~sim.pec_mask.T,
                                      np.ones_like(sim.pec_mask.T, dtype=float))
    ax.imshow(pec_display, origin='lower', cmap='Greys', alpha=0.85,
              aspect='equal', vmin=0, vmax=1)

    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    for step in range(n_steps):
        sim.step()
        if step % frame_skip == 0:
            current_max = max(np.abs(sim.Ez).max(), 0.01)
            norm = TwoSlopeNorm(vmin=-current_max, vcenter=0, vmax=current_max)
            im.set_data(sim.Ez.T)
            im.set_norm(norm)
            ax.set_title(f"{title}  [t={step}]", fontsize=11)
            plt.pause(0.001)

    energy = sim.compute_backscatter_energy()
    return energy


def evaluate_wall_quick(params, fdtd_cfg, n_segments=16,
                         wall_height=50, wall_thickness=4):
    """Évaluation rapide (sans animation) pour le GA."""
    sim = FDTD2D_TMz(fdtd_cfg)
    sim.set_wall_from_params(params, n_segments, wall_height, wall_thickness)
    sim.run()
    return sim.compute_backscatter_energy()
