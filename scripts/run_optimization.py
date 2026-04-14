#!/usr/bin/env python3
"""
================================================================================
PROJET PRINCIPAL : Optimisation de Géométrie de Mur Anti-Radar
             par Simulation FDTD 2D + Machine Learning
================================================================================

Ce script orchestre l'ensemble du pipeline :

1. Configuration de la simulation FDTD 2D TMz
2. Simulation de référence (mur plat) → mesure baseline de la RCS
3. Optimisation par Algorithme Génétique (GA)
4. Optimisation par CMA-ES (warm start depuis GA)
5. Optimisation par Reinforcement Learning (RL)
6. Comparaison des résultats et visualisation

Paramètres calibrés pour Apple M4 (10 cœurs, 16 Go RAM).
================================================================================
"""

import sys
import os

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

from src.fdtd import FDTD2D_TMz, FDTDConfig
from src.optim import GeneticAlgorithm, GAConfig, CMAES, CMAConfig
from src.optim.rl_agent import RLOptimizer, RLConfig
from src.optim.fitness import evaluate_wall, sim_counter
from src.viz import (
    plot_field_snapshot, plot_field_snapshots_grid,
    plot_wall_profile, plot_ga_convergence, plot_rl_convergence,
    plot_comparison, plot_project_summary
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

# Apple M4 : 8 workers (laisse 2 cœurs pour le système)
N_WORKERS = 8

# GA : population calibrée pour 8 workers (pop_size multiple de N_WORKERS)
GA_CFG = GAConfig(
    n_genes=N_SEGMENTS, pop_size=40, n_generations=80,
    crossover_rate=0.85, eta_c=10.0,
    eta_m=20.0, adaptive_mutation=True,
    elite_count=5, tournament_size=3,
    n_workers=N_WORKERS,
)

# CMA-ES : warm start depuis GA, affinement parallèle
CMA_CFG = CMAConfig(
    n_params=N_SEGMENTS, sigma0=0.4, max_iter=60,
    pop_size=24, param_min=-1.0, param_max=1.0, n_workers=N_WORKERS,
)

# RL : plus d'épisodes grâce au gain en temps sur GA/CMA-ES
RL_CFG = RLConfig(
    n_params=N_SEGMENTS, n_episodes=60, steps_per_episode=5,
    learning_rate=0.005, gamma=0.95, action_std_init=0.4,
    action_std_min=0.05, std_decay=0.99, n_rollouts=4,
)

# 3 angles pour robustesse angulaire (exploite le parallélisme)
INCIDENCE_ANGLES = [0.0, 15.0, -15.0]


# ==============================================================================
# Fonction de fitness (wrapper)
# ==============================================================================

def _evaluate_wall(params: np.ndarray) -> float:
    return evaluate_wall(params, FDTD_CFG, N_SEGMENTS, WALL_HEIGHT,
                         WALL_THICKNESS, INCIDENCE_ANGLES)


# ==============================================================================
# Phases du pipeline
# ==============================================================================

def run_baseline():
    print("\n" + "=" * 70)
    print("PHASE 1 : Simulation de référence (mur plat)")
    print("=" * 70)

    flat_params = np.zeros(N_SEGMENTS)

    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(flat_params, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)

    snapshots = []
    snapshot_times = [50, 150, 250, 349]

    for step in range(FDTD_CFG.n_steps):
        sim.step()
        if step in snapshot_times:
            ez_phys, pec_phys = sim.get_physical_fields()
            snapshots.append((ez_phys, pec_phys, f't = {step} Δt'))

    energy = sim.compute_backscatter_energy()

    print(f"  Énergie rétrodiffusée (mur plat) : {energy:.6f}")
    print(f"  Grille : {FDTD_CFG.nx}x{FDTD_CFG.ny} (+ PML {FDTD_CFG.n_pml} cellules)")
    print(f"  Fréquence : {FDTD_CFG.freq/1e9:.1f} GHz")
    print(f"  Longueur d'onde : {FDTD_CFG.wavelength*100:.2f} cm")
    print(f"  Pas spatial : {FDTD_CFG.dx*1e3:.3f} mm")
    print(f"  Pas temporel : {FDTD_CFG.dt*1e12:.3f} ps")

    print("  Génération des figures...")
    ez_phys, pec_phys = sim.get_physical_fields()
    plot_field_snapshots_grid(snapshots,
                              save_path=os.path.join(OUTPUT_DIR, "01_baseline_snapshots.png"))
    plot_field_snapshot(ez_phys, pec_phys,
                        title=f"Champ Ez - Mur plat (t={FDTD_CFG.n_steps}Δt)",
                        save_path=os.path.join(OUTPUT_DIR, "02_baseline_final.png"))

    return energy, sim


def run_ga_optimization():
    print("\n" + "=" * 70)
    print("PHASE 2 : Optimisation par Algorithme Génétique")
    print("=" * 70)

    ga = GeneticAlgorithm(GA_CFG, _evaluate_wall)
    best = ga.run(verbose=True)

    plot_ga_convergence(ga.history,
                        save_path=os.path.join(OUTPUT_DIR, "03_ga_convergence.png"))
    plot_wall_profile(best.genome,
                       title=f"Profil GA optimal (RCS={best.fitness:.4f})",
                       save_path=os.path.join(OUTPUT_DIR, "04_ga_profile.png"))

    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(best.genome, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)
    sim.run()

    ez_phys, pec_phys = sim.get_physical_fields()
    plot_field_snapshot(ez_phys, pec_phys,
                        title="Champ Ez - Mur GA optimal",
                        save_path=os.path.join(OUTPUT_DIR, "05_ga_field.png"))

    return best, ga.history


def run_cma_optimization(ga_best_genome=None):
    print("\n" + "=" * 70)
    print("PHASE 2b : Optimisation par CMA-ES")
    print("=" * 70)

    cma = CMAES(CMA_CFG, _evaluate_wall)
    best_params = cma.run(x0=ga_best_genome, verbose=True)

    plot_ga_convergence(cma.history,
                        save_path=os.path.join(OUTPUT_DIR, "03b_cma_convergence.png"))
    plot_wall_profile(best_params,
                       title=f"Profil CMA-ES optimal (RCS={cma.best_fitness:.4f})",
                       save_path=os.path.join(OUTPUT_DIR, "04b_cma_profile.png"))

    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(best_params, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)
    sim.run()

    ez_phys, pec_phys = sim.get_physical_fields()
    plot_field_snapshot(ez_phys, pec_phys,
                        title="Champ Ez - Mur CMA-ES optimal",
                        save_path=os.path.join(OUTPUT_DIR, "05b_cma_field.png"))

    return best_params, cma.best_fitness, cma.history


def run_rl_optimization():
    print("\n" + "=" * 70)
    print("PHASE 3 : Optimisation par Reinforcement Learning")
    print("=" * 70)

    rl = RLOptimizer(RL_CFG, _evaluate_wall)
    best_genome = rl.run(verbose=True)

    plot_rl_convergence(rl.history,
                        save_path=os.path.join(OUTPUT_DIR, "06_rl_convergence.png"))
    plot_wall_profile(best_genome,
                       title=f"Profil RL optimal (RCS={rl.best_fitness:.4f})",
                       save_path=os.path.join(OUTPUT_DIR, "07_rl_profile.png"))

    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(best_genome, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)
    sim.run()

    ez_phys, pec_phys = sim.get_physical_fields()
    plot_field_snapshot(ez_phys, pec_phys,
                        title="Champ Ez - Mur RL optimal",
                        save_path=os.path.join(OUTPUT_DIR, "08_rl_field.png"))

    return best_genome, rl.best_fitness, rl.history


def run_comparison(flat_fitness, ga_best, ga_history,
                   cma_genome, cma_fitness, cma_history,
                   rl_genome, rl_fitness, rl_history):
    print("\n" + "=" * 70)
    print("PHASE 5 : Comparaison et Synthèse")
    print("=" * 70)

    if ga_best.fitness <= cma_fitness:
        best_genome, best_fitness, best_name = ga_best.genome, ga_best.fitness, "GA"
    else:
        best_genome, best_fitness, best_name = cma_genome, cma_fitness, "CMA-ES"

    plot_comparison(
        best_genome, best_fitness,
        rl_genome, rl_fitness, flat_fitness,
        save_path=os.path.join(OUTPUT_DIR, "09_comparison.png")
    )

    best_history = ga_history if best_name == "GA" else cma_history
    plot_project_summary(
        FDTD_CFG, best_history, rl_history,
        best_genome, rl_genome, flat_fitness,
        save_path=os.path.join(OUTPUT_DIR, "10_summary.png")
    )

    from src.optim.fitness import sim_counter as sc
    results = [
        ("Mur plat (baseline)", flat_fitness),
        ("GA optimal", ga_best.fitness),
        ("CMA-ES optimal", cma_fitness),
        ("RL optimal", rl_fitness),
    ]

    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)
    for name, fit in results:
        reduction = (1 - fit / flat_fitness) * 100
        tag = f" (réduction: {reduction:.1f}%)" if name != "Mur plat (baseline)" else ""
        print(f"  {name:24s}: RCS = {fit:.6f}{tag}")

    best_method = min(results[1:], key=lambda x: x[1])
    print(f"\n  Total simulations FDTD : {sc}")
    print(f"  Meilleure méthode      : {best_method[0]}")
    print("=" * 70)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  OPTIMISATION DE GÉOMÉTRIE DE MUR ANTI-RADAR                     ║")
    print("║  FDTD 2D TMz + GA + CMA-ES + RL  - Apple M4 (10 cœurs)         ║")
    print("╠════════════════════════════════════════════════════════════════════╣")
    print(f"║  Fréquence radar : {FDTD_CFG.freq/1e9:.0f} GHz (bande X)                         ║")
    print(f"║  Longueur d'onde : {FDTD_CFG.wavelength*100:.2f} cm                               ║")
    print(f"║  Grille FDTD     : {FDTD_CFG.nx}x{FDTD_CFG.ny} cellules                          ║")
    print(f"║  Segments du mur : {N_SEGMENTS}                                            ║")
    print(f"║  Workers         : {N_WORKERS}                                             ║")
    print(f"║  Angles          : {INCIDENCE_ANGLES}                        ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    t0 = time.time()

    flat_fitness, baseline_sim = run_baseline()
    ga_best, ga_history = run_ga_optimization()
    cma_genome, cma_fitness, cma_history = run_cma_optimization(ga_best.genome)
    rl_genome, rl_fitness, rl_history = run_rl_optimization()
    run_comparison(flat_fitness, ga_best, ga_history,
                   cma_genome, cma_fitness, cma_history,
                   rl_genome, rl_fitness, rl_history)

    total_time = time.time() - t0
    print(f"\n  Temps total d'exécution : {total_time:.1f}s ({total_time/60:.1f} min)")

    print(f"\n  Fichiers générés dans {OUTPUT_DIR} :")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    - {f}")


if __name__ == "__main__":
    main()
