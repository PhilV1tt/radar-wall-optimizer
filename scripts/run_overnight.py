#!/usr/bin/env python3
"""
Optimisation GA overnight sur Mac Mini (10 cœurs).
Lance le GA avec une grande population pendant toute la nuit,
puis génère une animation MP4 du meilleur résultat.

Usage:
    cd ~/Documents/VSC/radar_wall_optimizer
    nohup python3 -u scripts/run_overnight.py > run_overnight.log 2>&1 &
"""

import multiprocessing as mp
mp.set_start_method('fork')  # macOS utilise 'spawn' par défaut, qui bloque sur le pickling

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

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import TwoSlopeNorm

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

# 5 angles d'incidence pour une robustesse angulaire maximale
INCIDENCE_ANGLES = [0.0, -10.0, 10.0, -20.0, 20.0]

N_WORKERS = 8  # 8 workers sur 10 cœurs (laisse 2 pour le système)

# GA overnight : grande population, toute la nuit
# ~0.17s/eval, pop=48, 5 angles → 48*5=240 evals/gen
# Avec 8 workers : ceil(48/8)*5*0.17 = 6*5*0.17 ≈ 5.1s/gen
# 8h = 28800s → ~5600 gens possibles, on met 5000
GA_CFG = GAConfig(
    n_genes=N_SEGMENTS, pop_size=48, n_generations=5000,
    crossover_rate=0.85, eta_c=10.0,
    eta_m=20.0, adaptive_mutation=True,
    elite_count=7, tournament_size=3,
    n_workers=N_WORKERS,
)


def _evaluate_wall(params: np.ndarray) -> float:
    return evaluate_wall(params, FDTD_CFG, N_SEGMENTS, WALL_HEIGHT,
                         WALL_THICKNESS, INCIDENCE_ANGLES)


# ==============================================================================
# Animation FDTD → MP4
# ==============================================================================

def save_fdtd_animation(params, filename, title="", n_steps=350, fps=30):
    """Simule la FDTD et sauvegarde une animation MP4."""
    print(f"  Génération de l'animation ({n_steps} steps)...")

    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(params, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)

    # Pré-calculer tous les frames
    frame_skip = 2
    frames_ez = []
    for step in range(n_steps):
        sim.step()
        if step % frame_skip == 0:
            frames_ez.append(sim.Ez.copy())

    pec_mask = sim.pec_mask.copy()

    # Créer la figure
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    fig.set_facecolor('black')
    ax.set_facecolor('black')

    vmax = max(np.abs(f).max() for f in frames_ez) * 0.8
    if vmax < 0.01:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    im = ax.imshow(frames_ez[0].T, origin='lower', cmap='RdBu_r',
                   norm=norm, aspect='equal', interpolation='bilinear')

    pec_display = np.ma.masked_where(~pec_mask.T,
                                      np.ones_like(pec_mask.T, dtype=float))
    ax.imshow(pec_display, origin='lower', cmap='Greys', alpha=0.85,
              aspect='equal', vmin=0, vmax=1)

    ax.set_title(title, fontsize=13, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    fig.colorbar(im, ax=ax, label='Ez', shrink=0.8)

    def update(frame_idx):
        ez = frames_ez[frame_idx]
        current_max = max(np.abs(ez).max(), 0.01)
        new_norm = TwoSlopeNorm(vmin=-current_max, vcenter=0, vmax=current_max)
        im.set_data(ez.T)
        im.set_norm(new_norm)
        step = frame_idx * frame_skip
        ax.set_title(f"{title}  [t={step}Δt]", fontsize=13,
                     fontweight='bold', color='white')
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=len(frames_ez),
                                    interval=1000 // fps, blit=True)

    writer = animation.FFMpegWriter(fps=fps, bitrate=2000)
    anim.save(filename, writer=writer, dpi=120)
    plt.close(fig)
    print(f"  Animation sauvegardée : {filename}")


# ==============================================================================
# Pipeline
# ==============================================================================

def main():
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  OPTIMISATION GA OVERNIGHT - MAC MINI 10 CŒURS                    ║")
    print("╠════════════════════════════════════════════════════════════════════╣")
    print(f"║  Fréquence  : {FDTD_CFG.freq/1e9:.0f} GHz | Grille : {FDTD_CFG.nx}x{FDTD_CFG.ny}               ║")
    print(f"║  Population : {GA_CFG.pop_size} | Générations : {GA_CFG.n_generations}                  ║")
    print(f"║  Angles     : {INCIDENCE_ANGLES}               ║")
    print(f"║  Workers    : {N_WORKERS}                                            ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    t0 = time.time()

    # --- Baseline ---
    print("\n" + "=" * 70)
    print("PHASE 1 : Baseline (mur plat)")
    print("=" * 70)

    flat_params = np.zeros(N_SEGMENTS)
    flat_fitness = _evaluate_wall(flat_params)
    print(f"  RCS mur plat ({len(INCIDENCE_ANGLES)} angles) : {flat_fitness:.6f}")

    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(flat_params, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)
    snapshots = []
    for step in range(FDTD_CFG.n_steps):
        sim.step()
        if step in [50, 150, 250, 349]:
            ez_phys, pec_phys = sim.get_physical_fields()
            snapshots.append((ez_phys, pec_phys, f't = {step} Δt'))

    ez_phys, pec_phys = sim.get_physical_fields()
    plot_field_snapshots_grid(snapshots,
                              save_path=os.path.join(OUTPUT_DIR, "01_baseline_snapshots.png"))
    plot_field_snapshot(ez_phys, pec_phys,
                        title="Champ Ez - Mur plat",
                        save_path=os.path.join(OUTPUT_DIR, "02_baseline_final.png"))

    # --- GA ---
    print("\n" + "=" * 70)
    print(f"PHASE 2 : Algorithme Génétique - {GA_CFG.pop_size} pop, "
          f"{GA_CFG.n_generations} gens, {N_WORKERS} workers")
    print("=" * 70)

    ga = GeneticAlgorithm(GA_CFG, _evaluate_wall)
    best = ga.run(verbose=True)

    plot_ga_convergence(ga.history,
                        save_path=os.path.join(OUTPUT_DIR, "03_ga_convergence.png"))
    plot_wall_profile(best.genome,
                       title=f"Profil GA optimal (RCS={best.fitness:.4f})",
                       save_path=os.path.join(OUTPUT_DIR, "04_ga_profile.png"))

    # --- Visualisation du meilleur mur ---
    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(best.genome, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)
    snapshots = []
    for step in range(FDTD_CFG.n_steps):
        sim.step()
        if step in [50, 150, 250, 349]:
            ez_phys, pec_phys = sim.get_physical_fields()
            snapshots.append((ez_phys, pec_phys, f't = {step} Δt'))

    ez_phys, pec_phys = sim.get_physical_fields()
    plot_field_snapshot(ez_phys, pec_phys,
                        title="Champ Ez - Mur GA optimal",
                        save_path=os.path.join(OUTPUT_DIR, "05_ga_field.png"))
    plot_field_snapshots_grid(snapshots,
                              save_path=os.path.join(OUTPUT_DIR, "05b_ga_snapshots.png"))

    # --- Animations MP4 ---
    print("\n" + "=" * 70)
    print("PHASE 3 : Animations")
    print("=" * 70)

    save_fdtd_animation(
        flat_params,
        os.path.join(OUTPUT_DIR, "anim_baseline.mp4"),
        title="Mur plat (baseline)",
    )

    save_fdtd_animation(
        best.genome,
        os.path.join(OUTPUT_DIR, "anim_ga_optimal.mp4"),
        title=f"Mur GA optimal (RCS={best.fitness:.4f})",
    )

    # --- Résultats ---
    total_time = time.time() - t0
    reduction = (1 - best.fitness / flat_fitness) * 100

    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"  Mur plat (baseline)  : RCS = {flat_fitness:.6f}")
    print(f"  GA optimal           : RCS = {best.fitness:.6f}")
    print(f"  Réduction RCS        : {reduction:.1f}%")
    print(f"  Angles d'incidence   : {INCIDENCE_ANGLES}")
    print(f"  Générations          : {GA_CFG.n_generations}")
    print(f"  Population           : {GA_CFG.pop_size}")
    print(f"  Temps total          : {total_time:.1f}s ({total_time/60:.1f} min / {total_time/3600:.1f}h)")
    print(f"  Meilleur génome      : {np.array2string(best.genome, precision=4)}")
    print("=" * 70)

    np.savez(os.path.join(OUTPUT_DIR, "best_ga_result.npz"),
             genome=best.genome, fitness=best.fitness,
             flat_fitness=flat_fitness,
             angles=INCIDENCE_ANGLES,
             history_best=ga.history['best_fitness'],
             history_mean=ga.history['mean_fitness'])

    print(f"\n  Fichiers générés dans {OUTPUT_DIR}/ :")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size = os.path.getsize(fpath)
        print(f"    - {f} ({size // 1024} KB)")


if __name__ == "__main__":
    main()
