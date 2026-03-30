#!/usr/bin/env python3
"""
Rapport physique complet — Radar Wall Optimizer
================================================

Génère un rapport de physique L3 avec :
  1. Configuration physique de la simulation (unités SI)
  2. Baseline mur plat : énergie rétrodiffusée + RCS bistatique (NTFF)
  3. Référence théorique (Optique Physique 2D)
  4. Optimisation par Algorithme Génétique
  5. Comparaison flat vs optimisé : réduction en dB
  6. Figures scientifiques (champs en mm, diagramme polaire bistatique)
  7. Fichier rapport.txt structuré

Métriques utilisées
-------------------
- Énergie rétrodiffusée E = ΣEz² (région SF devant le mur)  → métrique GA,
  réduction en dB = 10·log10(E_flat / E_opt)
- RCS bistatique NTFF (relatif) → diagramme polaire,
  montre la redistribution angulaire du champ diffusé

Usage
-----
  python run.py rapport
  python run.py rapport --seed 42 --out results/mon_rapport
  python scripts/run_rapport.py --seed 42 --convergence
"""

import argparse
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.fdtd import FDTD2D_TMz, FDTDConfig
from src.fdtd.ntff import analytical_rcs_flat_strip_2d
from src.optim.genetic import GeneticAlgorithm, GAConfig
from src.optim.fitness import evaluate_wall
from src.utils.console import console, info, success, warn


# ──────────────────────────────────────────────────────────────────────────────
# Paramètres par défaut du rapport
# ──────────────────────────────────────────────────────────────────────────────

RAPPORT_NX        = 100
RAPPORT_NY        = 100
RAPPORT_N_STEPS   = 400
RAPPORT_PPW       = 15
RAPPORT_N_SEGS    = 14
RAPPORT_WALL_H    = 40
RAPPORT_WALL_T    = 4
RAPPORT_POP       = 32
RAPPORT_N_GEN     = 60
RAPPORT_N_WORKERS = max(1, (os.cpu_count() or 2) // 2)

# Position du mur pour nx=100 (2/3 de la grille physique)
RAPPORT_WALL_CX   = 65
RAPPORT_WALL_CY   = 50


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_outdir(base: str | None) -> Path:
    if base:
        d = Path(base)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = Path("results") / f"rapport_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_cfg() -> FDTDConfig:
    """Crée la configuration FDTD pour le rapport."""
    return FDTDConfig(
        nx=RAPPORT_NX, ny=RAPPORT_NY,
        ppw=RAPPORT_PPW, freq=10e9,
        n_steps=RAPPORT_N_STEPS,
        wall_center_x=RAPPORT_WALL_CX,
        wall_center_y=RAPPORT_WALL_CY,
    )


def _run_simulation(cfg: FDTDConfig, params: np.ndarray,
                    n_segments: int, wall_height: int, wall_thickness: int):
    """Lance une simulation FDTD complète et retourne (sim, ez_phys, pec_phys)."""
    sim = FDTD2D_TMz(cfg)
    sim.set_wall_from_params(params, n_segments=n_segments,
                              wall_height=wall_height, wall_thickness=wall_thickness)
    sim.run()
    ez_phys, pec_phys = sim.get_physical_fields()
    return sim, ez_phys, pec_phys


# ──────────────────────────────────────────────────────────────────────────────
# Étape 1 : Baseline mur plat
# ──────────────────────────────────────────────────────────────────────────────

def step_baseline(cfg: FDTDConfig, wall_height: int, wall_thickness: int,
                  n_segments: int, with_plots: bool, outdir: Path):
    """Simule le mur plat de référence et retourne les métriques physiques."""
    info("Étape 1/3 — Baseline mur plat")

    params_flat = np.zeros(n_segments)
    sim, ez_flat, pec_flat = _run_simulation(
        cfg, params_flat, n_segments, wall_height, wall_thickness
    )

    energy_flat = float(sim.compute_backscatter_energy())
    angles_flat, rcs_bistatic_flat = sim.compute_bistatic_rcs(n_angles=180)

    # Référence analytique (Optique Physique 2D, Balanis §11.3)
    wall_height_m = wall_height * cfg.dx
    rcs_po = analytical_rcs_flat_strip_2d(wall_height_m, cfg.wavelength)

    console.print(f"  [cyan]Énergie rétrodiffusée (mur plat) :[/]  {energy_flat:.4e}")
    console.print(f"  [cyan]Hauteur mur w                    :[/]  {wall_height_m*1e3:.2f} mm = {wall_height_m/cfg.wavelength:.2f} λ")
    console.print(f"  [cyan]Référence PO  σ/λ = 2w²/λ²       :[/]  {rcs_po:.4f}")

    if with_plots:
        from src.viz.plots import plot_field_physical
        plot_field_physical(
            ez_flat, pec_flat, dx_mm=cfg.dx * 1e3,
            title=r"Champ $E_z$ — Mur plat (référence)",
            save_path=str(outdir / "field_flat.png"),
        )
        success(f"  Champ mur plat  → {outdir / 'field_flat.png'}")

    return {
        "energy": energy_flat,
        "rcs_po": rcs_po,
        "angles": angles_flat,
        "rcs_bistatic": rcs_bistatic_flat,
        "ez": ez_flat,
        "pec": pec_flat,
        "wall_height_m": wall_height_m,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Étape 2 : Optimisation GA
# ──────────────────────────────────────────────────────────────────────────────

def step_ga(cfg: FDTDConfig, n_segments: int, wall_height: int,
            wall_thickness: int, n_gen: int, pop_size: int,
            n_workers: int, seed: int | None, time_budget: float | None):
    """Lance l'algorithme génétique et retourne (best, ga)."""
    info("Étape 2/3 — Optimisation par Algorithme Génétique")

    def fitness(params):
        return evaluate_wall(params, cfg, n_segments=n_segments,
                             wall_height=wall_height, wall_thickness=wall_thickness)

    ga_cfg = GAConfig(
        n_genes=n_segments,
        pop_size=pop_size,
        n_generations=n_gen,
        n_workers=n_workers,
        seed=seed,
        elite_count=3,
        n_hall_of_fame=8,
        stagnation_window=15,
        time_budget=time_budget,
    )
    ga = GeneticAlgorithm(ga_cfg, fitness)
    best = ga.run(verbose=True)

    console.print(f"  [green]Meilleur fitness GA (énergie) :[/]  {best.fitness:.6e}")
    return best, ga


# ──────────────────────────────────────────────────────────────────────────────
# Étape 3 : Post-évaluation du meilleur profil GA
# ──────────────────────────────────────────────────────────────────────────────

def step_post_eval(cfg: FDTDConfig, best_genome: np.ndarray,
                   n_segments: int, wall_height: int, wall_thickness: int,
                   with_plots: bool, outdir: Path):
    """Post-évalue le meilleur profil GA : énergie + RCS bistatique."""
    info("Étape 3/3 — Post-évaluation du meilleur profil GA")

    sim, ez_opt, pec_opt = _run_simulation(
        cfg, best_genome, n_segments, wall_height, wall_thickness
    )

    energy_opt = float(sim.compute_backscatter_energy())
    angles_opt, rcs_bistatic_opt = sim.compute_bistatic_rcs(n_angles=180)

    console.print(f"  [green]Énergie rétrodiffusée (optimisé) :[/]  {energy_opt:.4e}")

    if with_plots:
        from src.viz.plots import plot_field_physical
        plot_field_physical(
            ez_opt, pec_opt, dx_mm=cfg.dx * 1e3,
            title=r"Champ $E_z$ — Mur optimisé (GA)",
            save_path=str(outdir / "field_optimal.png"),
        )
        success(f"  Champ mur optimal → {outdir / 'field_optimal.png'}")

    return {
        "energy": energy_opt,
        "angles": angles_opt,
        "rcs_bistatic": rcs_bistatic_opt,
        "ez": ez_opt,
        "pec": pec_opt,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Figures de synthèse
# ──────────────────────────────────────────────────────────────────────────────

def generate_figures(baseline, opt, ga, cfg: FDTDConfig, outdir: Path):
    """Génère les figures scientifiques de synthèse."""
    from src.viz.plots import plot_rcs_polar_comparison, plot_ga_convergence

    # Diagramme polaire bistatique (comparaison relative valide)
    plot_rcs_polar_comparison(
        baseline["angles"],
        baseline["rcs_bistatic"],
        opt["rcs_bistatic"],
        save_path=str(outdir / "rcs_polar.png"),
    )
    success(f"  Diagramme polaire → {outdir / 'rcs_polar.png'}")

    # Convergence GA
    plot_ga_convergence(
        ga.history,
        save_path=str(outdir / "convergence_ga.png"),
    )
    success(f"  Convergence GA    → {outdir / 'convergence_ga.png'}")


# ──────────────────────────────────────────────────────────────────────────────
# Études de convergence optionnelles
# ──────────────────────────────────────────────────────────────────────────────

def run_convergence_studies(cfg: FDTDConfig, wall_height: int,
                             wall_thickness: int, outdir: Path):
    """Lance les études de convergence grille et PML."""
    from src.validation.convergence import grid_convergence_study, pml_convergence_study
    from src.viz.plots import plot_convergence_study, plot_pml_convergence

    info("Étude de convergence — résolution spatiale")
    ppw_list = [8, 10, 12, 15, 20]
    ppw_vals, energy_grid = grid_convergence_study(
        ppw_list,
        base_nx=cfg.nx, base_ny=cfg.ny,
        base_n_steps=cfg.n_steps, freq=cfg.freq,
        n_pml=cfg.n_pml,
        wall_height=wall_height, wall_thickness=wall_thickness,
    )
    plot_convergence_study(ppw_vals, energy_grid,
                           save_path=str(outdir / "convergence_grid.png"))
    success(f"  Convergence grille → {outdir / 'convergence_grid.png'}")

    info("Étude de convergence — épaisseur PML")
    n_pml_list = [6, 8, 10, 12, 16]
    pml_vals, rcs_pml = pml_convergence_study(
        n_pml_list,
        ppw=cfg.ppw, base_nx=cfg.nx, base_ny=cfg.ny,
        base_n_steps=cfg.n_steps, freq=cfg.freq,
        wall_height=wall_height, wall_thickness=wall_thickness,
    )
    dx_mm = cfg.dx * 1e3
    plot_pml_convergence(pml_vals, rcs_pml, dx_mm=dx_mm,
                         save_path=str(outdir / "convergence_pml.png"))
    success(f"  Convergence PML    → {outdir / 'convergence_pml.png'}")

    return {
        "ppw_list": ppw_vals, "energy_grid": energy_grid,
        "n_pml_list": pml_vals, "energy_pml": rcs_pml,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Rapport texte
# ──────────────────────────────────────────────────────────────────────────────

def write_rapport(cfg: FDTDConfig, baseline: dict, opt: dict,
                  best_genome: np.ndarray, ga_history: dict,
                  n_segments: int, wall_height: int, wall_thickness: int,
                  seed: int | None, outdir: Path,
                  convergence: dict | None = None):
    """Écrit le fichier rapport.txt structuré."""
    energy_flat = baseline["energy"]
    energy_opt = opt["energy"]
    reduction_db = 10.0 * math.log10(max(energy_flat, 1e-30) / max(energy_opt, 1e-30))
    reduction_pct = (1.0 - energy_opt / max(energy_flat, 1e-30)) * 100.0
    wall_height_mm = baseline["wall_height_m"] * 1e3
    wall_wl = baseline["wall_height_m"] / cfg.wavelength

    lines = [
        "=" * 70,
        "  RAPPORT — OPTIMISATION DE MUR ANTI-RADAR PAR ALGORITHME GÉNÉTIQUE",
        "=" * 70,
        "",
        f"  Date           : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"  Graine RNG     : {seed if seed is not None else 'aléatoire'}",
        f"  Dossier sortie : {outdir}",
        "",
        "── 1. CONFIGURATION PHYSIQUE ────────────────────────────────────────",
        "",
        cfg.physical_summary(),
        f"  Mur PEC        : hauteur = {wall_height} cellules = {wall_height_mm:.2f} mm",
        f"                   épaisseur = {wall_thickness} cellules",
        f"                   segments de contrôle = {n_segments}",
        f"  Position mur   : x = {cfg.wall_center_x} (physique), "
        f"soit {cfg.wall_center_x * cfg.dx * 1e3:.1f} mm depuis le bord",
        "",
        "── 2. MODÈLE PHYSIQUE ────────────────────────────────────────────────",
        "",
        "  Simulation : FDTD 2D en polarisation TMz (Ez, Hx, Hy non nuls).",
        "  Équations de Maxwell discrétisées sur grille de Yee (schéma leapfrog).",
        "  Source : wavelet de Ricker centrée à f₀, injectée par technique TFSF.",
        "  Absorbing BC : CPML (Convolutional PML) sur les 4 bords.",
        "  Analyse : NTFF (transformée champ proche → champ lointain) pour",
        "            le diagramme bistatique de RCS.",
        "",
        "  Métrique d'optimisation : E_retro = ∑Ez²  dans la zone SF devant",
        "  le mur (mesure l'énergie rétrodiffusée vers le radar).",
        "",
        "── 3. RÉFÉRENCE THÉORIQUE (Optique Physique 2D) ─────────────────────",
        "",
        "  Pour un mur PEC plan (incidence normale, polarisation TMz) :",
        "    σ_2D / λ = 2w² / λ²  (Balanis, Advanced Engineering EM, §11.3)",
        "",
        f"  Hauteur mur w  : {wall_height_mm:.2f} mm = {wall_wl:.2f} λ",
        f"  RCS théorique  : σ_2D/λ = {baseline['rcs_po']:.4f}",
        "",
        "  Note : la RCS NTFF absolue ne peut pas être directement comparée",
        "  à cette formule (calibration des unités FDTD). En revanche, la",
        "  RÉDUCTION relative (flat → optimisé) est une grandeur fiable.",
        "",
        "── 4. BASELINE — MUR PLAT ────────────────────────────────────────────",
        "",
        f"  Énergie rétrodiffusée : E_flat = {energy_flat:.6e}  (u.a.)",
        "",
        "── 5. OPTIMISATION — ALGORITHME GÉNÉTIQUE ───────────────────────────",
        "",
        f"  Nombre de générations : {len(ga_history['best_fitness'])}",
        f"  Taille population     : {RAPPORT_POP}",
        f"  Fitness initiale (gen 1)  : {ga_history['best_fitness'][0]:.6e}",
        f"  Fitness finale (gen {len(ga_history['best_fitness']):3d}) : "
        f"{ga_history['best_fitness'][-1]:.6e}",
        "",
        "── 6. RÉSULTAT — MEILLEUR PROFIL GA ─────────────────────────────────",
        "",
        f"  Énergie rétrodiffusée : E_opt = {energy_opt:.6e}  (u.a.)",
        "",
        f"  RÉDUCTION DE RCS      : {reduction_db:+.2f} dB  ({reduction_pct:.1f} %)",
        f"  Interprétation        : le mur optimisé réduit l'énergie renvoyée",
        f"  vers le radar d'un facteur {energy_flat/max(energy_opt,1e-30):.2f}×",
        f"  ({reduction_db:.1f} dB).",
        "",
        "  Paramètres du profil optimal :",
        "  " + "  ".join(f"{v:+.3f}" for v in best_genome),
        "",
    ]

    if convergence is not None:
        lines += [
            "── 7. ÉTUDES DE CONVERGENCE ─────────────────────────────────────────",
            "",
            "  Convergence en résolution spatiale (énergie rétrodiffusée) :",
        ]
        for ppw, e in zip(convergence["ppw_list"], convergence["energy_grid"]):
            lines.append(f"    ppw={ppw:2d}  E_retro={e:.4e}")
        lines += [
            "",
            "  Convergence en épaisseur PML :",
        ]
        for n_pml, e in zip(convergence["n_pml_list"], convergence["energy_pml"]):
            lines.append(f"    n_pml={n_pml:2d}  E_retro={e:.4e}")
        lines.append("")

    lines += [
        "── FIGURES GÉNÉRÉES ─────────────────────────────────────────────────",
        "",
        "  field_flat.png      — Champ Ez mur plat (axes en mm)",
        "  field_optimal.png   — Champ Ez mur optimisé (axes en mm)",
        "  rcs_polar.png       — SER bistatique (dB relatif) : flat vs GA",
        "  convergence_ga.png  — Courbe de convergence du GA",
    ]
    if convergence is not None:
        lines += [
            "  convergence_grid.png — Énergie vs résolution spatiale",
            "  convergence_pml.png  — Énergie vs épaisseur PML",
        ]
    lines += [
        "",
        "── RÉFÉRENCES ───────────────────────────────────────────────────────",
        "",
        "  [1] K.S. Yee, \"Numerical solution of initial boundary value problems",
        "      involving Maxwell's equations in isotropic media\", IEEE TAP, 1966.",
        "  [2] A. Taflove & S. Hagness, Computational Electrodynamics, 3e éd., 2005.",
        "  [3] C.A. Balanis, Advanced Engineering Electromagnetics, 2e éd., 2012.",
        "  [4] K. Deb & R.B. Agrawal, \"Simulated binary crossover\", 1995.",
        "  [5] J.-P. Berenger, \"A perfectly matched layer for absorption\", JCP, 1994.",
        "",
        "=" * 70,
    ]

    rapport_path = outdir / "rapport.txt"
    rapport_path.write_text("\n".join(lines), encoding="utf-8")
    success(f"  Rapport texte     → {rapport_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Rapport physique — Radar Wall Optimizer")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--out",         type=str,   default=None)
    parser.add_argument("--no-plots",    action="store_true")
    parser.add_argument("--convergence", action="store_true",
                        help="Lance les études de convergence (+10 min)")
    parser.add_argument("--time",        type=float, default=None, metavar="MINUTES",
                        help="Budget temps GA en minutes")
    parser.add_argument("--workers",     type=int,   default=None)
    args = parser.parse_args()

    t0 = time.time()
    outdir = _make_outdir(args.out)
    n_workers = args.workers or RAPPORT_N_WORKERS
    time_budget = args.time * 60 if args.time else None

    console.rule("[bold blue]Radar Wall Optimizer — Rapport L3[/]")
    info(f"Graine RNG : {args.seed}")
    info(f"Dossier    : {outdir}")
    info(f"Workers    : {n_workers}")

    cfg = _make_cfg()

    console.rule("[bold]Configuration physique[/]")
    console.print(cfg.physical_summary())

    # Étape 1 — Baseline
    console.rule("[bold]Étape 1 / 3 — Baseline[/]")
    baseline = step_baseline(
        cfg, RAPPORT_WALL_H, RAPPORT_WALL_T, RAPPORT_N_SEGS,
        with_plots=not args.no_plots, outdir=outdir,
    )

    # Étape 2 — GA
    console.rule("[bold]Étape 2 / 3 — Algorithme Génétique[/]")
    best, ga = step_ga(
        cfg, RAPPORT_N_SEGS, RAPPORT_WALL_H, RAPPORT_WALL_T,
        n_gen=RAPPORT_N_GEN, pop_size=RAPPORT_POP,
        n_workers=n_workers, seed=args.seed,
        time_budget=time_budget,
    )

    # Étape 3 — Post-évaluation
    console.rule("[bold]Étape 3 / 3 — Post-évaluation NTFF[/]")
    opt = step_post_eval(
        cfg, best.genome, RAPPORT_N_SEGS, RAPPORT_WALL_H, RAPPORT_WALL_T,
        with_plots=not args.no_plots, outdir=outdir,
    )

    # Calcul réduction
    energy_flat = baseline["energy"]
    energy_opt = opt["energy"]
    reduction_db = 10.0 * math.log10(max(energy_flat, 1e-30) / max(energy_opt, 1e-30))

    # Figures de synthèse
    if not args.no_plots:
        console.rule("[bold]Figures[/]")
        generate_figures(baseline, opt, ga, cfg, outdir)

    # Études de convergence (optionnel)
    convergence = None
    if args.convergence:
        console.rule("[bold]Études de convergence[/]")
        convergence = run_convergence_studies(
            cfg, RAPPORT_WALL_H, RAPPORT_WALL_T, outdir
        )

    # Rapport texte
    console.rule("[bold]Rapport[/]")
    write_rapport(
        cfg, baseline, opt, best.genome, ga.history,
        RAPPORT_N_SEGS, RAPPORT_WALL_H, RAPPORT_WALL_T,
        seed=args.seed, outdir=outdir, convergence=convergence,
    )

    # Sauvegarde npz
    np.savez(
        outdir / "rapport_results.npz",
        genome=best.genome,
        fitness_ga=np.array([best.fitness]),
        energy_flat=np.array([energy_flat]),
        energy_opt=np.array([energy_opt]),
        reduction_db=np.array([reduction_db]),
        rcs_po=np.array([baseline["rcs_po"]]),
        history_best=np.array(ga.history["best_fitness"]),
        history_mean=np.array(ga.history["mean_fitness"]),
    )
    success(f"  Données           → {outdir / 'rapport_results.npz'}")

    elapsed = time.time() - t0

    console.rule("[bold green]Résumé final[/]")
    console.print(f"  Énergie mur plat  : [cyan]{energy_flat:.4e}[/]")
    console.print(f"  Énergie optimisé  : [green]{energy_opt:.4e}[/]")
    console.print(f"  Réduction         : [bold green]{reduction_db:+.2f} dB[/]")
    console.print(f"  Réf. PO (théorie) : σ/λ = {baseline['rcs_po']:.3f} "
                  f"(w = {baseline['wall_height_m']/cfg.wavelength:.2f} λ)")
    console.print(f"  Durée totale      : {elapsed/60:.1f} min")
    console.print(f"  Dossier           : {outdir}")


if __name__ == "__main__":
    main()
