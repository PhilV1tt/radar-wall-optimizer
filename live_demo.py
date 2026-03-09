#!/usr/bin/env python3
"""
================================================================================
VISUALISATION EN TEMPS RÉEL — FDTD + Optimisation Génétique
================================================================================

Affiche en temps réel :
  - Panneau gauche  : simulation FDTD 2D (onde + mur)
  - Panneau haut-droit : profil du mur actuel vs meilleur trouvé
  - Panneau bas-droit : courbe de convergence du GA

Utilisation : python3 live_demo.py
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Backend interactif pour macOS
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
import time
import sys

# Imports du projet
from fdtd_2d import FDTD2D_TMz, FDTDConfig, C0
from genetic_algorithm import GeneticAlgorithm, GAConfig


# ==============================================================================
# Configuration
# ==============================================================================

N_SEGMENTS = 16
WALL_HEIGHT = 50
WALL_THICKNESS = 4

FDTD_CFG = FDTDConfig(
    nx=150, ny=150, ppw=15, freq=10e9,
    courant=0.5, n_steps=350,
    tfsf_margin=12, wall_center_x=95, wall_center_y=75,
)

GA_CFG = GAConfig(
    n_genes=N_SEGMENTS, pop_size=1000, n_generations=2,
    mutation_rate=0.2, mutation_sigma=0.35,
    crossover_rate=0.85, elite_fraction=0.15, tournament_size=3,
)


# ==============================================================================
# Simulation FDTD avec animation
# ==============================================================================

def run_fdtd_animated(params, ax, title="", n_steps=300, frame_skip=5):
    """Lance une simulation FDTD et anime le champ Ez en temps réel."""
    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(params, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)

    # Premier affichage
    vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(sim.Ez.T, origin='lower', cmap='RdBu_r', norm=norm,
                   aspect='equal', interpolation='bilinear')

    # Masque PEC
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


def evaluate_wall_quick(params):
    """Évaluation rapide (sans animation) pour le GA."""
    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(params, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)
    sim.run()
    return sim.compute_backscatter_energy()


# ==============================================================================
# Boucle principale avec affichage live
# ==============================================================================

def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DEMO TEMPS RÉEL — Optimisation de mur anti-radar          ║")
    print("║  Ferme la fenêtre matplotlib pour arrêter                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # --- Setup figure ---
    plt.ion()
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.4, 1],
                           hspace=0.35, wspace=0.3)

    ax_field = fig.add_subplot(gs[:, 0])   # FDTD (grande, à gauche)
    ax_profile = fig.add_subplot(gs[0, 1]) # Profil du mur (haut-droit)
    ax_conv = fig.add_subplot(gs[1, 1])    # Convergence (bas-droit)

    # Style sombre
    for ax in [ax_field, ax_profile, ax_conv]:
        ax.set_facecolor('#16213e')
        ax.tick_params(colors='white', labelsize=8)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_color('#333')

    fig.suptitle('Optimisation de Mur Anti-Radar — Temps Réel',
                 fontsize=16, fontweight='bold', color='white')

    plt.show()

    # ==================================================================
    # Phase 1 : Mur plat (baseline) — avec animation
    # ==================================================================
    ax_field.clear()
    ax_field.set_facecolor('#16213e')
    flat_energy = run_fdtd_animated(
        np.zeros(N_SEGMENTS), ax_field,
        title="Phase 1 : Mur plat (baseline)", n_steps=300, frame_skip=4
    )
    ax_field.set_title(f"Mur plat — RCS = {flat_energy:.1f}", fontsize=11,
                       fontweight='bold', color='white')
    plt.pause(1.0)

    # ==================================================================
    # Phase 2 : Optimisation GA avec visualisation live
    # ==================================================================
    best_fitnesses = []
    mean_fitnesses = []
    best_genome = np.zeros(N_SEGMENTS)
    best_fitness = flat_energy

    # Init profil plot
    y_seg = np.linspace(0, 1, N_SEGMENTS)

    ga = GeneticAlgorithm(GA_CFG, evaluate_wall_quick)
    ga.initialize_population()
    ga.evaluate_population()
    ga.population.sort(key=lambda ind: ind.fitness)
    ga.best_individual = ga.population[0]
    best_genome = ga.best_individual.genome.copy()
    best_fitness = ga.best_individual.fitness

    for gen in range(GA_CFG.n_generations):
        t0 = time.time()

        # --- Évoluer ---
        ga.evolve_one_generation(gen)
        current_best = ga.best_individual

        best_fitnesses.append(current_best.fitness)
        mean_fitnesses.append(ga.history['mean_fitness'][-1])

        dt = time.time() - t0
        reduction = (1 - current_best.fitness / flat_energy) * 100

        print(f"Gen {gen+1:2d}/{GA_CFG.n_generations} | "
              f"Best: {current_best.fitness:8.1f} | "
              f"Mean: {mean_fitnesses[-1]:8.1f} | "
              f"Réduction: {reduction:5.1f}% | {dt:.1f}s")

        # --- Update profil (haut-droit) ---
        ax_profile.clear()
        ax_profile.set_facecolor("#20305a")

        # Mur plat en référence
        ax_profile.plot(np.zeros(N_SEGMENTS), y_seg, '--',
                        color='#555', linewidth=1, label='Plat')

        # Meilleur profil
        ax_profile.fill_betweenx(y_seg,
                                  current_best.genome,
                                  current_best.genome + 0.12,
                                  color='#00d2ff', alpha=0.4)
        ax_profile.plot(current_best.genome, y_seg, '-',
                        color='#00d2ff', linewidth=2.5, label='Meilleur')

        ax_profile.set_xlim(-1.3, 1.3)
        ax_profile.set_xlabel('Déplacement', color='white')
        ax_profile.set_ylabel('Position', color='white')
        ax_profile.set_title(f'Profil du mur — Gen {gen+1}',
                             color='white', fontsize=11)
        ax_profile.legend(loc='upper right', fontsize=8,
                          facecolor='#16213e', edgecolor='#333',
                          labelcolor='white')
        ax_profile.tick_params(colors='white')

        # --- Update convergence (bas-droit) ---
        ax_conv.clear()
        ax_conv.set_facecolor('#16213e')

        gens_range = range(1, len(best_fitnesses) + 1)
        ax_conv.semilogy(gens_range, best_fitnesses, '-',
                         color='#00d2ff', linewidth=2.5, label='Best')
        ax_conv.semilogy(gens_range, mean_fitnesses, '--',
                         color='#ffa500', linewidth=1.5, alpha=0.7, label='Mean')
        ax_conv.axhline(y=flat_energy, color='#ff4444', linestyle=':',
                        alpha=0.5, label=f'Plat ({flat_energy:.0f})')

        ax_conv.set_xlabel('Génération', color='white')
        ax_conv.set_ylabel('RCS (énergie)', color='white')
        ax_conv.set_title(f'Convergence — Réduction: {reduction:.1f}%',
                          color='#00ff88', fontsize=11, fontweight='bold')
        ax_conv.legend(loc='upper right', fontsize=8,
                       facecolor='#16213e', edgecolor='#333',
                       labelcolor='white')
        ax_conv.tick_params(colors='white')
        ax_conv.grid(True, alpha=0.15, color='white')

        plt.pause(0.05)

    # ==================================================================
    # Phase 3 : Animation finale avec le mur optimal
    # ==================================================================
    print(f"\n  Animation finale avec le profil optimal...")
    ax_field.clear()
    ax_field.set_facecolor('#16213e')

    optimal_energy = run_fdtd_animated(
        current_best.genome, ax_field,
        title="Mur optimal GA", n_steps=350, frame_skip=3
    )

    ax_field.set_title(
        f"Mur optimal — RCS = {optimal_energy:.1f}  "
        f"(réduction {(1 - optimal_energy/flat_energy)*100:.0f}%)",
        fontsize=12, fontweight='bold', color='#00ff88'
    )

    print(f"\n  ✓ Terminé ! Réduction finale : {(1-optimal_energy/flat_energy)*100:.1f}%")
    print(f"  Ferme la fenêtre pour quitter.")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()