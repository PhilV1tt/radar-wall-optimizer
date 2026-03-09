import numpy as np
import os
import sys
import time

# Modules du projet
from fdtd_2d import FDTD2D_TMz, FDTDConfig, C0
from genetic_algorithm import GeneticAlgorithm, GAConfig
from rl_optimizer import RLOptimizer, RLConfig
from visualization import (
    plot_field_snapshot, plot_field_snapshots_grid,
    plot_wall_profile, plot_ga_convergence, plot_rl_convergence,
    plot_comparison, plot_project_summary
)


# ==============================================================================
# Configuration
# ==============================================================================

# Répertoire de sortie
OUTPUT_DIR = "./results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paramètres de la simulation FDTD
# On utilise une grille modérée pour que l'optimisation soit tractable
FDTD_CFG = FDTDConfig(
    nx=150,           # cellules en x
    ny=150,           # cellules en y
    ppw=15,           # 15 points par longueur d'onde (compromis vitesse/précision)
    freq=10e9,        # 10 GHz (bande X — radar de surveillance typique)
    courant=0.5,      # Nombre de Courant (< 1/√2 ≈ 0.707 pour stabilité 2D)
    n_steps=350,      # Assez pour que l'onde traverse le domaine et interagisse
    tfsf_margin=12,   # Marge TFSF en cellules
    wall_center_x=95, # Position du mur (2/3 du domaine)
    wall_center_y=75,  # Centre vertical
)

# Paramètres du GA
N_SEGMENTS = 16  # Nombre de segments de contrôle du profil

GA_CFG = GAConfig(
    n_genes=N_SEGMENTS,
    pop_size=20,        # Population réduite pour la démo
    n_generations=25,   # Générations réduites pour la démo
    mutation_rate=0.2,
    mutation_sigma=0.35,
    crossover_rate=0.85,
    elite_fraction=0.15,
    tournament_size=3,
)

# Paramètres du RL
RL_CFG = RLConfig(
    n_params=N_SEGMENTS,
    n_episodes=40,
    steps_per_episode=4,
    learning_rate=0.005,
    gamma=0.95,
    action_std_init=0.4,
    action_std_min=0.05,
    std_decay=0.99,
    n_rollouts=3,
)

# Paramètres du mur
WALL_HEIGHT = 50    # cellules
WALL_THICKNESS = 4  # cellules


# ==============================================================================
# Fonction de fitness (wrapper FDTD)
# ==============================================================================

# Compteur global de simulations
sim_counter = 0

def evaluate_wall(params: np.ndarray) -> float:
    """Évalue la RCS d'un profil de mur via simulation FDTD.
    
    C'est la fonction coûteuse du pipeline : chaque évaluation nécessite
    une simulation FDTD complète (~350 pas de temps).
    
    Paramètres
    ----------
    params : np.ndarray
        Vecteur de N_SEGMENTS valeurs dans [-1, 1] définissant le profil.
    
    Returns
    -------
    fitness : float
        Énergie du champ rétrodiffusé (proxy pour la RCS).
        Plus c'est petit, mieux le mur diffuse l'onde.
    """
    global sim_counter
    sim_counter += 1
    
    # Créer et configurer la simulation
    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(params, n_segments=N_SEGMENTS,
                              wall_height=WALL_HEIGHT, 
                              wall_thickness=WALL_THICKNESS)
    
    # Lancer la simulation
    sim.run()
    
    # Mesurer l'énergie rétrodiffusée
    energy = sim.compute_backscatter_energy()
    
    return energy


# ==============================================================================
# Simulation de référence
# ==============================================================================

def run_baseline():
    """Simule un mur plat comme référence."""
    print("\n" + "=" * 70)
    print("PHASE 1 : Simulation de référence (mur plat)")
    print("=" * 70)
    
    flat_params = np.zeros(N_SEGMENTS)
    
    # Simulation
    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(flat_params, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)
    
    # Capturer des snapshots
    snapshots = []
    snapshot_times = [50, 150, 250, 349]
    
    for step in range(FDTD_CFG.n_steps):
        sim.step()
        if step in snapshot_times:
            snapshots.append((
                sim.Ez.copy(), 
                sim.pec_mask.copy(),
                f't = {step} Δt'
            ))
    
    # Mesures
    energy = sim.compute_backscatter_energy()
    
    print(f"  Énergie rétrodiffusée (mur plat) : {energy:.6f}")
    print(f"  Grille : {FDTD_CFG.nx}×{FDTD_CFG.ny}")
    print(f"  Fréquence : {FDTD_CFG.freq/1e9:.1f} GHz")
    print(f"  Longueur d'onde : {FDTD_CFG.wavelength*100:.2f} cm")
    print(f"  Pas spatial : {FDTD_CFG.dx*1e3:.3f} mm")
    print(f"  Pas temporel : {FDTD_CFG.dt*1e12:.3f} ps")
    
    # Visualisation
    print("  Génération des figures...")
    plot_field_snapshots_grid(snapshots, 
                              save_path=os.path.join(OUTPUT_DIR, "01_baseline_snapshots.png"))
    
    plot_field_snapshot(sim.Ez, sim.pec_mask,
                        title=f"Champ Ez — Mur plat (t={FDTD_CFG.n_steps}Δt)",
                        save_path=os.path.join(OUTPUT_DIR, "02_baseline_final.png"))
    
    return energy, sim


# ==============================================================================
# Optimisation par GA
# ==============================================================================

def run_ga_optimization():
    """Lance l'optimisation par algorithme génétique."""
    print("\n" + "=" * 70)
    print("PHASE 2 : Optimisation par Algorithme Génétique")
    print("=" * 70)
    
    ga = GeneticAlgorithm(GA_CFG, evaluate_wall)
    best = ga.run(verbose=True)
    
    # Visualisation de la convergence
    plot_ga_convergence(ga.history,
                        save_path=os.path.join(OUTPUT_DIR, "03_ga_convergence.png"))
    
    # Profil optimal
    plot_wall_profile(best.genome, 
                       title=f"Profil GA optimal (RCS={best.fitness:.4f})",
                       save_path=os.path.join(OUTPUT_DIR, "04_ga_profile.png"))
    
    # Simulation finale avec le profil optimal
    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(best.genome, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)
    sim.run()
    
    plot_field_snapshot(sim.Ez, sim.pec_mask,
                        title=f"Champ Ez — Mur GA optimal",
                        save_path=os.path.join(OUTPUT_DIR, "05_ga_field.png"))
    
    return best, ga.history


# ==============================================================================
# Optimisation par RL
# ==============================================================================

def run_rl_optimization():
    """Lance l'optimisation par reinforcement learning."""
    print("\n" + "=" * 70)
    print("PHASE 3 : Optimisation par Reinforcement Learning")
    print("=" * 70)
    
    rl = RLOptimizer(RL_CFG, evaluate_wall)
    best_genome = rl.run(verbose=True)
    
    # Visualisation
    plot_rl_convergence(rl.history,
                        save_path=os.path.join(OUTPUT_DIR, "06_rl_convergence.png"))
    
    plot_wall_profile(best_genome,
                       title=f"Profil RL optimal (RCS={rl.best_fitness:.4f})",
                       save_path=os.path.join(OUTPUT_DIR, "07_rl_profile.png"))
    
    # Simulation finale
    sim = FDTD2D_TMz(FDTD_CFG)
    sim.set_wall_from_params(best_genome, N_SEGMENTS, WALL_HEIGHT, WALL_THICKNESS)
    sim.run()
    
    plot_field_snapshot(sim.Ez, sim.pec_mask,
                        title=f"Champ Ez — Mur RL optimal",
                        save_path=os.path.join(OUTPUT_DIR, "08_rl_field.png"))
    
    return best_genome, rl.best_fitness, rl.history


# ==============================================================================
# Comparaison et synthèse
# ==============================================================================

def run_comparison(flat_fitness, ga_best, ga_history, rl_genome, rl_fitness, rl_history):
    """Compare les résultats GA vs RL vs baseline."""
    print("\n" + "=" * 70)
    print("PHASE 4 : Comparaison et Synthèse")
    print("=" * 70)
    
    # Comparaison des profils
    plot_comparison(
        ga_best.genome, ga_best.fitness,
        rl_genome, rl_fitness,
        flat_fitness,
        save_path=os.path.join(OUTPUT_DIR, "09_comparison.png")
    )
    
    # Figure de synthèse
    plot_project_summary(
        FDTD_CFG, ga_history, rl_history,
        ga_best.genome, rl_genome, flat_fitness,
        save_path=os.path.join(OUTPUT_DIR, "10_summary.png")
    )
    
    # Résumé textuel
    print("\n" + "=" * 70)
    print("RÉSULTATS FINAUX")
    print("=" * 70)
    print(f"  Mur plat (baseline)    : RCS = {flat_fitness:.6f}")
    print(f"  GA optimal             : RCS = {ga_best.fitness:.6f} "
          f"(réduction: {(1-ga_best.fitness/flat_fitness)*100:.1f}%)")
    print(f"  RL optimal             : RCS = {rl_fitness:.6f} "
          f"(réduction: {(1-rl_fitness/flat_fitness)*100:.1f}%)")
    print(f"\n  Total simulations FDTD : {sim_counter}")
    print(f"  Meilleure méthode      : {'GA' if ga_best.fitness < rl_fitness else 'RL'}")
    print("=" * 70)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Point d'entrée principal."""
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  OPTIMISATION DE GÉOMÉTRIE DE MUR ANTI-RADAR                     ║")
    print("║  FDTD 2D TMz + Algorithme Génétique + Reinforcement Learning     ║")
    print("╠════════════════════════════════════════════════════════════════════╣")
    print(f"║  Fréquence radar : {FDTD_CFG.freq/1e9:.0f} GHz (bande X)                         ║")
    print(f"║  Longueur d'onde : {FDTD_CFG.wavelength*100:.2f} cm                               ║")
    print(f"║  Grille FDTD     : {FDTD_CFG.nx}×{FDTD_CFG.ny} cellules                          ║")
    print(f"║  Segments du mur : {N_SEGMENTS}                                            ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    
    t0 = time.time()
    
    # Phase 1 : Baseline
    flat_fitness, baseline_sim = run_baseline()
    
    # Phase 2 : GA
    ga_best, ga_history = run_ga_optimization()
    
    # Phase 3 : RL
    rl_genome, rl_fitness, rl_history = run_rl_optimization()
    
    # Phase 4 : Comparaison
    run_comparison(flat_fitness, ga_best, ga_history, rl_genome, rl_fitness, rl_history)
    
    total_time = time.time() - t0
    print(f"\n  Temps total d'exécution : {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Liste des fichiers générés
    print(f"\n  Fichiers générés dans {OUTPUT_DIR} :")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"    • {f}")


if __name__ == "__main__":
    main()
