#!/usr/bin/env python3
"""
Tests de validation physique — convergence grille et absorption PML.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.fdtd import FDTD2D_TMz, FDTDConfig


def test_grid_convergence():
    """Test de convergence en raffinant la grille (ppw = 10, 15, 20, 25, 30)."""
    print("=" * 60)
    print("Test de convergence grille")
    print("=" * 60)

    ppw_values = [10, 15, 20, 25, 30]
    energies = []

    for ppw in ppw_values:
        cfg = FDTDConfig(
            nx=100, ny=100, ppw=ppw, freq=10e9, courant=0.5,
            n_steps=300, n_pml=10, tfsf_margin=10,
            wall_center_x=60, wall_center_y=50,
        )
        sim = FDTD2D_TMz(cfg)
        sim.set_wall_from_params(np.zeros(10), n_segments=10,
                                  wall_height=30, wall_thickness=4)
        sim.run()
        energy = sim.compute_backscatter_energy()
        energies.append(energy)
        print(f"  ppw={ppw:2d}  dx={cfg.dx*1e3:.3f} mm  energy={energy:.4f}")

    # Vérifier la convergence : les valeurs doivent se stabiliser
    rel_changes = [abs(energies[i] - energies[i-1]) / max(energies[i-1], 1e-10)
                   for i in range(1, len(energies))]
    print(f"\n  Changements relatifs : {[f'{c:.3f}' for c in rel_changes]}")
    print(f"  Convergence OK : {'oui' if rel_changes[-1] < 0.2 else 'non (>20%)'}")


def test_pml_absorption():
    """Test d'absorption PML — énergie résiduelle après propagation."""
    print("\n" + "=" * 60)
    print("Test d'absorption PML")
    print("=" * 60)

    cfg = FDTDConfig(
        nx=80, ny=80, ppw=15, freq=10e9, courant=0.5,
        n_steps=100, n_pml=12, tfsf_margin=10,
        wall_center_x=50, wall_center_y=40,
    )
    sim = FDTD2D_TMz(cfg)
    n = cfg.n_pml

    # Phase 1 : pulse
    sim.run(150)
    energy_mid = np.sum(sim.Ez[n:-n, n:-n]**2)

    # Phase 2 : absorption
    sim.run(200)
    energy_late = np.sum(sim.Ez[n:-n, n:-n]**2)

    ratio = energy_late / max(energy_mid, 1e-30)
    print(f"  Énergie après pulse   : {energy_mid:.6f}")
    print(f"  Énergie après absorption : {energy_late:.6f}")
    print(f"  Ratio                 : {ratio:.6f}")
    print(f"  Absorption OK         : {'oui' if ratio < 0.1 else 'non (>10%)'}")


if __name__ == "__main__":
    test_grid_convergence()
    test_pml_absorption()
    print("\n  Validation terminée.")
