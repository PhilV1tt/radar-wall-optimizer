#!/usr/bin/env python3
"""
Radar Wall Optimizer — point d'entrée unifié.

Presets
-------
  fast      ~3 min   GA rapide (feedback dev, petite grille)
  medium    ~20 min  GA complet + visualisations
  full      ~40 min  GA + CMA-ES + RL  (délègue à scripts/run_optimization.py)
  validate           Tests physiques FDTD (convergence en grille + PML)

Usage
-----
  python run.py fast
  python run.py medium --workers 8 --seed 42
  python run.py medium --time 40          # GA pendant 40 min, arrêt propre
  python run.py full
  python run.py validate

Options
-------
  --workers N     Nombre de workers parallèles (défaut : auto = cpu//2)
  --seed N        Graine aléatoire (reproductibilité)
  --out DIR       Dossier de sortie  (défaut : results/YYYYMMDD_HHMMSS/)
  --no-plots      Ne pas générer les graphiques matplotlib
  --checkpoint N  Sauvegarder un checkpoint tous les N générations (défaut : 25)
  --time MINUTES  Budget temps en minutes (arrête le GA proprement à la fin
                  de la génération courante). Compatible avec fast et medium.
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Résolution des imports depuis la racine du projet ─────────────────────────
sys.path.insert(0, str(Path(__file__).parent))


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _auto_workers() -> int:
    cpus = os.cpu_count() or 2
    return max(1, cpus // 2)


def _make_outdir(base: str | None) -> Path:
    if base:
        d = Path(base)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        d = Path("results") / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def _make_checkpoint_fn(outdir: Path, every: int):
    """Retourne un callback qui sauvegarde le meilleur génome toutes les `every` gen."""
    def _fn(gen: int, best):
        if gen % every == 0:
            path = outdir / f"checkpoint_gen{gen:04d}.npz"
            np.savez(path, genome=best.genome, fitness=np.array([best.fitness]))
    return _fn


# ──────────────────────────────────────────────────────────────────────────────
# Presets
# ──────────────────────────────────────────────────────────────────────────────

def _fdtd_cfg(nx, ny, n_steps, ppw=12):
    from src.fdtd import FDTDConfig
    return FDTDConfig(nx=nx, ny=ny, n_steps=n_steps, ppw=ppw)


def preset_fast(args):
    """GA rapide sur petite grille — ~3 min."""
    from src.utils.console import console, info, success
    from src.optim.genetic import GeneticAlgorithm, GAConfig
    from src.optim.fitness import evaluate_wall

    outdir = _make_outdir(args.out)
    workers = args.workers or min(4, _auto_workers())
    info(f"Preset [bold]fast[/] · workers={workers} · out={outdir}")

    fdtd_cfg = _fdtd_cfg(nx=100, ny=100, n_steps=300)
    n_segments = 12

    def fitness(params):
        return evaluate_wall(params, fdtd_cfg, n_segments=n_segments,
                             wall_height=40, wall_thickness=4)

    time_budget = args.time * 60 if args.time else None
    n_gen = int(1e6) if time_budget else 40
    ga_cfg = GAConfig(
        n_genes=n_segments,
        pop_size=24,
        n_generations=n_gen,
        n_workers=workers,
        seed=args.seed,
        elite_count=2,
        n_hall_of_fame=5,
        stagnation_window=15,
        time_budget=time_budget,
    )
    ga = GeneticAlgorithm(ga_cfg, fitness)
    best = ga.run(verbose=True,
                  checkpoint_fn=_make_checkpoint_fn(outdir, args.checkpoint))

    _save_result(outdir, best, ga, "fast")
    if not args.no_plots:
        _plot_ga(outdir, ga, fdtd_cfg, best, n_segments, "fast")

    return best


def preset_medium(args):
    """GA complet + visualisations — ~20 min."""
    from src.utils.console import console, info, success
    from src.optim.genetic import GeneticAlgorithm, GAConfig
    from src.optim.fitness import evaluate_wall

    outdir = _make_outdir(args.out)
    workers = args.workers or _auto_workers()
    info(f"Preset [bold]medium[/] · workers={workers} · out={outdir}")

    fdtd_cfg = _fdtd_cfg(nx=130, ny=130, n_steps=400)
    n_segments = 16
    incidence = [0.0, 15.0 * np.pi / 180, -15.0 * np.pi / 180]

    def fitness(params):
        return evaluate_wall(params, fdtd_cfg, n_segments=n_segments,
                             wall_height=50, wall_thickness=4,
                             incidence_angles=incidence)

    time_budget = args.time * 60 if args.time else None
    n_gen = int(1e6) if time_budget else 100
    ga_cfg = GAConfig(
        n_genes=n_segments,
        pop_size=48,
        n_generations=n_gen,
        n_workers=workers,
        seed=args.seed,
        elite_count=3,
        n_hall_of_fame=10,
        stagnation_window=20,
        time_budget=time_budget,
    )
    ga = GeneticAlgorithm(ga_cfg, fitness)
    best = ga.run(verbose=True,
                  checkpoint_fn=_make_checkpoint_fn(outdir, args.checkpoint))

    _save_result(outdir, best, ga, "medium")
    if not args.no_plots:
        _plot_ga(outdir, ga, fdtd_cfg, best, n_segments, "medium")

    return best


def preset_full(args):
    """Pipeline complet GA + CMA-ES + RL — délègue à scripts/run_optimization.py."""
    from src.utils.console import info, warn
    script = Path(__file__).parent / "scripts" / "run_optimization.py"
    if not script.exists():
        warn(f"Script introuvable : {script}")
        sys.exit(1)
    info(f"Lancement de [bold]{script.name}[/] …")
    os.execv(sys.executable, [sys.executable, str(script)])


def preset_validate(args):
    """Tests physiques FDTD — ~5 min."""
    from src.utils.console import info
    script = Path(__file__).parent / "scripts" / "run_validation.py"
    if not script.exists():
        info("Script run_validation.py introuvable.")
        sys.exit(1)
    os.execv(sys.executable, [sys.executable, str(script)])


def preset_rapport(args):
    """Rapport physique complet : validation + optimisation GA + figures — ~15 min."""
    from src.utils.console import info, warn
    script = Path(__file__).parent / "scripts" / "run_rapport.py"
    if not script.exists():
        warn(f"Script introuvable : {script}")
        sys.exit(1)
    info(f"Lancement de [bold]{script.name}[/] …")
    extra = ["--seed", str(args.seed)] if args.seed is not None else ["--seed", "42"]
    if args.out:
        extra += ["--out", args.out]
    if args.no_plots:
        extra.append("--no-plots")
    if args.workers:
        extra += ["--workers", str(args.workers)]
    if args.time:
        extra += ["--time", str(args.time)]
    os.execv(sys.executable, [sys.executable, str(script)] + extra)


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde & visualisation
# ──────────────────────────────────────────────────────────────────────────────

def _save_result(outdir: Path, best, ga, tag: str):
    from src.utils.console import success
    path = outdir / f"best_{tag}.npz"
    np.savez(path,
             genome=best.genome,
             fitness=np.array([best.fitness]),
             history_best=np.array(ga.history["best_fitness"]),
             history_mean=np.array(ga.history["mean_fitness"]))
    success(f"Résultat sauvegardé → {path}")


def _plot_ga(outdir: Path, ga, fdtd_cfg, best, n_segments: int, tag: str):
    """Génère les graphiques de convergence et de champ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.utils.console import success, info
    from src.fdtd import FDTD2D_TMz

    # 1. Courbe de convergence
    fig, ax = plt.subplots(figsize=(8, 4))
    gens = range(len(ga.history["best_fitness"]))
    ax.semilogy(gens, ga.history["best_fitness"], label="Meilleur (HOF)", lw=2)
    ax.semilogy(gens, ga.history["mean_fitness"], label="Moyenne pop.", lw=1, alpha=0.6)
    ax.set_xlabel("Génération")
    ax.set_ylabel("Fitness (énergie rétrodiffusée)")
    ax.set_title(f"Convergence GA — preset {tag}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = outdir / f"convergence_{tag}.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    success(f"Convergence → {p}")

    # 2. Champ électrique optimal
    try:
        sim = FDTD2D_TMz(fdtd_cfg)
        sim.set_wall_from_params(best.genome, n_segments=n_segments)
        sim.run()
        ez, pec = sim.get_physical_fields()

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        im = axes[0].imshow(ez.T, origin="lower", cmap="RdBu_r",
                            vmin=-ez.std() * 3, vmax=ez.std() * 3)
        axes[0].contour(pec.T, levels=[0.5], colors="k", linewidths=1)
        axes[0].set_title("Champ Ez (mur optimal)")
        fig.colorbar(im, ax=axes[0], fraction=0.046)

        axes[1].imshow(pec.T, origin="lower", cmap="Greys")
        axes[1].set_title("Géométrie du mur (PEC)")

        fig.suptitle(f"Profil optimal — preset {tag}  |  fitness={best.fitness:.3e}")
        fig.tight_layout()
        p = outdir / f"field_{tag}.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        success(f"Champ      → {p}")
    except Exception as e:
        from src.utils.console import warn
        warn(f"Impossible de générer le champ : {e}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Radar Wall Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "preset",
        choices=["fast", "medium", "full", "validate", "rapport"],
        help="Preset de configuration",
    )
    parser.add_argument("--workers", type=int, default=None,
                        help="Nombre de workers (défaut : cpu//2)")
    parser.add_argument("--seed",    type=int, default=None,
                        help="Graine aléatoire")
    parser.add_argument("--out",     type=str, default=None,
                        help="Dossier de sortie (défaut : results/YYYYMMDD_HHMMSS/)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Ne pas générer les graphiques")
    parser.add_argument("--checkpoint", type=int, default=25, metavar="N",
                        help="Sauvegarder un checkpoint tous les N générations (défaut : 25)")
    parser.add_argument("--time", type=float, default=None, metavar="MINUTES",
                        help="Budget temps en minutes (arrêt propre à la fin de la génération)")
    args = parser.parse_args()

    dispatch = {
        "fast":     preset_fast,
        "medium":   preset_medium,
        "full":     preset_full,
        "validate": preset_validate,
        "rapport":  preset_rapport,
    }
    dispatch[args.preset](args)


if __name__ == "__main__":
    main()
