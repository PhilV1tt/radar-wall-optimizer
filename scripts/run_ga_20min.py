#!/usr/bin/env python3
"""
Optimisation GA uniquement — ~20 min sur serveur 32 cœurs.
Toute la puissance de calcul est dédiée à l'algorithme génétique.
"""

import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

from src.fdtd import FDTD2D_TMz, FDTDConfig
from src.optim import GeneticAlgorithm, GAConfig
from src.optim.fitness import evaluate_wall
from src.viz import (
    plot_field_snapshot, plot_field_snapshots_grid,
    plot_wall_profile, plot_ga_convergence,
)

# ==============================================================================
# Configuration
# ==============================================================================

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FDTD_CFG = FDTDConfig(
    nx=150, ny=150, ppw=15, freq=10e9, courant=0.5,
    n_steps=350, tfsf_margin=12, wall_center_x=95, wall_center_y=75,
)

N_SEGMENTS = 16
WALL_HEIGHT = 50
WALL_THICKNESS = 4

# 3 angles d'incidence pour la robustesse
INCIDENCE_ANGLES = [0.0, 15.0, -15.0]

N_WORKERS = 16

# GA massif : pop=64, ~125 générations en 20 min
# Chaque gen ≈ 9.5s (64 individus, 3 angles, 16 workers)
GA_CFG = GAConfig(
    n_genes=N_SEGMENTS, pop_size=64, n_generations=125,
    crossover_rate=0.85, eta_c=10.0,
    eta_m=20.0, adaptive_mutation=True,
    elite_count=10, tournament_size=3,
    n_workers=N_WORKERS,
)


def _evaluate_wall(params: np.ndarray) -> float:
    return evaluate_wall(params, FDTD_CFG, N_SEGMENTS, WALL_HEIGHT,
                         WALL_THICKNESS, INCIDENCE_ANGLES)


def run_baseline():
    print("\n" + "=" * 70)
    print("PHASE 1 : Simulation de référence (mur plat)")
    print("=" * 70)

    flat_params = np.zeros(N_SEGMENTS)
    flat_fitness = _evaluate_wall(flat_params)

    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(flat_params, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)

    snapshots = []
    snapshot_times = [50, 150, 250, 349]
    for step in range(FDTD_CFG.n_steps):
        sim.step()
        if step in snapshot_times:
            ez_phys, pec_phys = sim.get_physical_fields()
            snapshots.append((ez_phys, pec_phys, f't = {step} Δt'))

    print(f"  RCS mur plat ({len(INCIDENCE_ANGLES)} angles) : {flat_fitness:.6f}")
    print(f"  Grille : {FDTD_CFG.nx}x{FDTD_CFG.ny}")
    print(f"  Fréquence : {FDTD_CFG.freq/1e9:.1f} GHz")
    print(f"  Angles : {INCIDENCE_ANGLES}")

    ez_phys, pec_phys = sim.get_physical_fields()
    plot_field_snapshots_grid(snapshots,
                              save_path=os.path.join(OUTPUT_DIR, "01_baseline_snapshots.png"))
    plot_field_snapshot(ez_phys, pec_phys,
                        title=f"Champ Ez — Mur plat (t={FDTD_CFG.n_steps}Δt)",
                        save_path=os.path.join(OUTPUT_DIR, "02_baseline_final.png"))
    return flat_fitness


def run_ga():
    print("\n" + "=" * 70)
    print(f"PHASE 2 : Algorithme Génétique — {N_WORKERS} workers, {len(INCIDENCE_ANGLES)} angles")
    print(f"  Population: {GA_CFG.pop_size} | Générations: {GA_CFG.n_generations}")
    print("=" * 70)

    ga = GeneticAlgorithm(GA_CFG, _evaluate_wall)
    best = ga.run(verbose=True)

    plot_ga_convergence(ga.history,
                        save_path=os.path.join(OUTPUT_DIR, "03_ga_convergence.png"))
    plot_wall_profile(best.genome,
                       title=f"Profil GA optimal (RCS={best.fitness:.4f})",
                       save_path=os.path.join(OUTPUT_DIR, "04_ga_profile.png"))

    # Simulation finale pour visualisation
    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(best.genome, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)
    snapshots = []
    snapshot_times = [50, 150, 250, 349]
    for step in range(FDTD_CFG.n_steps):
        sim.step()
        if step in snapshot_times:
            ez_phys, pec_phys = sim.get_physical_fields()
            snapshots.append((ez_phys, pec_phys, f't = {step} Δt'))

    ez_phys, pec_phys = sim.get_physical_fields()
    plot_field_snapshot(ez_phys, pec_phys,
                        title="Champ Ez — Mur GA optimal",
                        save_path=os.path.join(OUTPUT_DIR, "05_ga_field.png"))
    plot_field_snapshots_grid(snapshots,
                              save_path=os.path.join(OUTPUT_DIR, "05b_ga_snapshots.png"))

    return best, ga.history


def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  OPTIMISATION GA — SERVEUR 32 CŒURS — ~20 min                    ║")
    print("╠════════════════════════════════════════════════════════════════════╣")
    print(f"║  Fréquence  : {FDTD_CFG.freq/1e9:.0f} GHz | Grille : {FDTD_CFG.nx}x{FDTD_CFG.ny}               ║")
    print(f"║  Population : {GA_CFG.pop_size} | Générations : {GA_CFG.n_generations}                    ║")
    print(f"║  Angles     : {INCIDENCE_ANGLES}                        ║")
    print(f"║  Workers    : {N_WORKERS}                                            ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    t0 = time.time()

    flat_fitness = run_baseline()
    best, history = run_ga()

    total_time = time.time() - t0
    reduction = (1 - best.fitness / flat_fitness) * 100

    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"  Mur plat (baseline)  : RCS = {flat_fitness:.6f}")
    print(f"  GA optimal           : RCS = {best.fitness:.6f}")
    print(f"  Réduction RCS        : {reduction:.1f}%")
    print(f"  Angles d'incidence   : {INCIDENCE_ANGLES}")
    print(f"  Temps total          : {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Meilleur génome      : {np.array2string(best.genome, precision=4)}")
    print("=" * 70)

    # Sauvegarder le meilleur résultat
    np.savez(os.path.join(OUTPUT_DIR, "best_ga_result.npz"),
             genome=best.genome, fitness=best.fitness,
             flat_fitness=flat_fitness,
             angles=INCIDENCE_ANGLES,
             history_best=history['best_fitness'],
             history_mean=history['mean_fitness'])
    print(f"\n  Résultats sauvegardés dans {OUTPUT_DIR}/best_ga_result.npz")

    print(f"\n  Fichiers générés :")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    - {f}")


if __name__ == "__main__":
    main()
