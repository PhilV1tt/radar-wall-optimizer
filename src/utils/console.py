"""
Affichage Rich pour le terminal.

Usage dans les autres modules :
    from src.utils.console import console, GADisplay, info, success, warn
"""
from __future__ import annotations
import math
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn, Progress, SpinnerColumn,
    TaskProgressColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    from src.optim.genetic import GAConfig, GeneticAlgorithm

console = Console(highlight=False)


# ── Helpers ───────────────────────────────────────────────────────────────────

def success(msg: str): console.print(f"[bold green]✓[/] {msg}")
def info(msg: str):    console.print(f"[bold blue]·[/] {msg}")
def warn(msg: str):    console.print(f"[bold yellow]⚠[/] {msg}")
def error(msg: str):   console.print(f"[bold red]✗[/] {msg}")


# ── Affichage GA ──────────────────────────────────────────────────────────────

class GADisplay:
    """Affichage Rich temps-réel pour l'algorithme génétique."""

    _HEADER = (
        f"  {'Gen':>5}  {'Best':>13}  {'Mean':>13}  {'Std':>11}  "
        f"{'Div':>7}  {'η_m':>7}  {'Succ':>6}  Info"
    )

    def __init__(self, cfg: "GAConfig"):
        self._cfg = cfg
        self._progress: Optional[Progress] = None
        self._task_id = None
        self._best = float("inf")

    def start(self):
        from src.utils.xp import BACKEND
        cfg = self._cfg
        parallel = f"{cfg.n_workers} workers" if cfg.n_workers > 1 else "séquentiel"
        mut = (f"PM adaptatif 1/5 · η_m={cfg.eta_m:.0f}"
               if cfg.adaptive_mutation else f"PM fixe · η_m={cfg.eta_m:.0f}")
        sel = (f"tournoi k={cfg.tournament_size}"
               if cfg.selection == "tournament"
               else f"rang s={cfg.rank_pressure}")

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", min_width=20)
        grid.add_column()
        for key, val in [
            ("Population",  f"{cfg.pop_size} individus · {cfg.n_generations} générations"),
            ("Gènes",       f"{cfg.n_genes} · bornes [{cfg.gene_min}, {cfg.gene_max}]"),
            ("Croisement",  f"SBX η_c={cfg.eta_c:.0f} · p_c={cfg.crossover_rate:.2f}"),
            ("Mutation",    f"{mut} · p_m={cfg.mutation_rate:.4f}"),
            ("Sélection",   sel),
            ("Évaluation",  parallel),
            ("Backend",     BACKEND),
        ]:
            grid.add_row(key, val)

        console.print(Panel(grid,
                            title="[bold white]GA · Optimisation mur anti-radar",
                            border_style="blue", expand=False))
        console.print(f"[dim]{self._HEADER}[/dim]")
        console.rule(style="dim blue")

        self._progress = Progress(
            SpinnerColumn(style="blue"),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=28, style="dim blue", complete_style="bold blue"),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TextColumn("·"),
            TimeRemainingColumn(),
            TextColumn("· [bold green]{task.fields[best]}"),
            console=console,
        )
        self._progress.start()
        self._task_id = self._progress.add_task(
            f"0/{cfg.n_generations}", total=cfg.n_generations, best="—",
        )

    def update(self, gen: int, history: dict, dt: float, restarted: bool):
        if not history["best_fitness"]:
            return

        best  = history["best_fitness"][-1]
        mean  = history["mean_fitness"][-1]
        std   = history["std_fitness"][-1]
        div   = history["diversity"][-1]
        eta_m = history["eta_m"][-1]
        sr    = history["success_rate"][-1]

        if best < self._best:
            self._best = best

        sr_str = f"{sr:.0%}" if not math.isnan(sr) else "    —"
        if restarted:
            info_str, style = "[bold yellow]RESTART[/bold yellow]", "yellow"
        elif gen == 0:
            info_str, style = "init", "dim"
        else:
            info_str, style = f"{dt:.1f}s", ""

        row = (
            f"  {gen:>5}  {best:>13.6e}  {mean:>13.6e}  {std:>11.3e}  "
            f"{div:>7.4f}  {eta_m:>7.1f}  {sr_str:>6}  {info_str}"
        )
        self._progress.console.print(f"[{style}]{row}[/{style}]" if style else row)
        self._progress.update(
            self._task_id,
            advance=1,
            description=f"{gen}/{self._cfg.n_generations}",
            best=f"{self._best:.3e}",
        )

    def finish(self, elapsed: float, ga: "GeneticAlgorithm"):
        self._progress.stop()
        console.rule(style="dim blue")

        best  = ga._hof.best
        cache = ga._eval_fn if hasattr(ga._eval_fn, "hit_rate") else None

        grid = Table.grid(padding=(0, 2))
        grid.add_column(style="bold cyan", min_width=20)
        grid.add_column()
        grid.add_row("Durée",        f"{elapsed:.1f}s  ({elapsed/60:.1f} min)")
        grid.add_row("Évaluations",  str(ga._n_evals))
        grid.add_row("Redémarrages", str(ga._n_restarts))
        if cache:
            grid.add_row("Cache",
                         f"{cache.hits} hits · {cache.misses} miss  ({cache.hit_rate:.1%})")
        if best:
            grid.add_row("Meilleure fitness",
                         f"[bold green]{best.fitness:.6e}[/]  (gen {best.generation})")

        console.print(Panel(grid, title="[bold white]Résultats GA",
                            border_style="green", expand=False))
