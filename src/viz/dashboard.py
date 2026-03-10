"""
Dashboard interactif — démo temps réel FDTD + GA.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time

from src.fdtd import FDTD2D_TMz, FDTDConfig
from src.optim import GeneticAlgorithm, GAConfig
from src.viz.animation import run_fdtd_animated, evaluate_wall_quick


def main():
    N_SEGMENTS = 16
    WALL_HEIGHT = 50
    WALL_THICKNESS = 4

    fdtd_cfg = FDTDConfig(
        nx=150, ny=150, ppw=15, freq=10e9,
        courant=0.5, n_steps=350,
        tfsf_margin=12, wall_center_x=95, wall_center_y=75,
    )

    ga_cfg = GAConfig(
        n_genes=N_SEGMENTS, pop_size=20, n_generations=30,
        crossover_rate=0.85, eta_c=10.0,
        eta_m=20.0, adaptive_mutation=True,
        elite_count=3, tournament_size=3,
    )

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  DEMO TEMPS RÉEL — Optimisation de mur anti-radar          ║")
    print("║  Ferme la fenêtre matplotlib pour arrêter                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    plt.ion()
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor('#1a1a2e')

    gs = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[1.4, 1],
                           hspace=0.35, wspace=0.3)

    ax_field = fig.add_subplot(gs[:, 0])
    ax_profile = fig.add_subplot(gs[0, 1])
    ax_conv = fig.add_subplot(gs[1, 1])

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

    # Phase 1 : Mur plat (baseline)
    ax_field.clear()
    ax_field.set_facecolor('#16213e')
    flat_energy = run_fdtd_animated(
        np.zeros(N_SEGMENTS), ax_field, fdtd_cfg,
        n_segments=N_SEGMENTS, wall_height=WALL_HEIGHT,
        wall_thickness=WALL_THICKNESS,
        title="Phase 1 : Mur plat (baseline)", n_steps=300, frame_skip=4
    )
    ax_field.set_title(f"Mur plat — RCS = {flat_energy:.1f}", fontsize=11,
                       fontweight='bold', color='white')
    plt.pause(1.0)

    # Phase 2 : Optimisation GA
    best_fitnesses = []
    mean_fitnesses = []
    y_seg = np.linspace(0, 1, N_SEGMENTS)

    def fitness_fn(params):
        return evaluate_wall_quick(params, fdtd_cfg,
                                    n_segments=N_SEGMENTS,
                                    wall_height=WALL_HEIGHT,
                                    wall_thickness=WALL_THICKNESS)

    ga = GeneticAlgorithm(ga_cfg, fitness_fn)
    ga.initialize_population()
    ga.evaluate_population()
    ga.population.sort(key=lambda ind: ind.fitness)
    ga.best_individual = ga.population[0]

    for gen in range(ga_cfg.n_generations):
        t0 = time.time()
        ga.evolve_one_generation(gen)
        current_best = ga.best_individual

        best_fitnesses.append(current_best.fitness)
        mean_fitnesses.append(ga.history['mean_fitness'][-1])

        dt = time.time() - t0
        reduction = (1 - current_best.fitness / flat_energy) * 100

        print(f"Gen {gen+1:2d}/{ga_cfg.n_generations} | "
              f"Best: {current_best.fitness:8.1f} | "
              f"Mean: {mean_fitnesses[-1]:8.1f} | "
              f"Réduction: {reduction:5.1f}% | {dt:.1f}s")

        # Update profil
        ax_profile.clear()
        ax_profile.set_facecolor('#16213e')
        ax_profile.plot(np.zeros(N_SEGMENTS), y_seg, '--',
                        color='#555', linewidth=1, label='Plat')
        ax_profile.fill_betweenx(y_seg, current_best.genome,
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

        # Update convergence
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

    # Phase 3 : Animation finale
    print(f"\n  Animation finale avec le profil optimal...")
    ax_field.clear()
    ax_field.set_facecolor('#16213e')

    optimal_energy = run_fdtd_animated(
        current_best.genome, ax_field, fdtd_cfg,
        n_segments=N_SEGMENTS, wall_height=WALL_HEIGHT,
        wall_thickness=WALL_THICKNESS,
        title="Mur optimal GA", n_steps=350, frame_skip=3
    )

    ax_field.set_title(
        f"Mur optimal — RCS = {optimal_energy:.1f}  "
        f"(réduction {(1 - optimal_energy/flat_energy)*100:.0f}%)",
        fontsize=12, fontweight='bold', color='#00ff88'
    )

    print(f"\n  Terminé ! Réduction finale : {(1-optimal_energy/flat_energy)*100:.1f}%")
    print(f"  Ferme la fenêtre pour quitter.")

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
