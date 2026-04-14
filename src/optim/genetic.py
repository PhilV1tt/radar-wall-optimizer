"""
Algorithme Génétique (GA) - implémentation scientifique.

Opérateurs
----------
Initialisation  : Latin Hypercube Sampling               [LHS]
Sélection       : Tournoi / Rang linéaire                [RK]
Croisement      : Simulated Binary Crossover (SBX)       [SBX]
Mutation        : Mutation Polynomiale bornée (PM)        [PM]
Adaptation σ    : Règle 1/5 de Rechenberg sur η_m        [1/5]
Diversité       : Métrique normalisée + restart adaptatif
Élitisme        : Hall of Fame (N meilleurs all-time)

Références
----------
[LHS] McKay, M.D., Beckman, R.J. & Conover, W.J. (1979). A comparison of
      three methods for selecting values of input variables in the analysis
      of output from a computer code. Technometrics, 21(2), 239-245.
[RK]  Baker, J.E. (1985). Adaptive selection methods for genetic algorithms.
      Proc. 1st International Conference on Genetic Algorithms, 101-111.
[SBX] Deb, K. & Agrawal, R.B. (1995). Simulated binary crossover for
      continuous search space. Complex Systems, 9(2), 115-148.
[PM]  Deb, K. & Goyal, M. (1996). A combined genetic adaptive search (GeneAS)
      for engineering design. Computer Science and Informatics, 26(4), 30-45.
[1/5] Rechenberg, I. (1973). Evolutionstrategie: Optimierung technischer
      Systeme nach Prinzipien der biologischen Evolution. Frommann-Holzboog.
"""

import hashlib
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np


# ==============================================================================
# Configuration
# ==============================================================================

@dataclass
class GAConfig:
    """Paramètres complets de l'algorithme génétique."""

    # ── Problème ───────────────────────────────────────────────────────────────
    n_genes: int = 20
    gene_min: float = -1.0
    gene_max: float = 1.0

    # ── Population ────────────────────────────────────────────────────────────
    pop_size: int = 50
    n_generations: int = 100
    seed: Optional[int] = None          # Graine pour la reproductibilité

    # ── Sélection ─────────────────────────────────────────────────────────────
    selection: str = "tournament"       # "tournament" | "rank"
    tournament_size: int = 3
    rank_pressure: float = 1.5          # Pression sélective s ∈ [1.0, 2.0]

    # ── Croisement - SBX (Deb & Agrawal, 1995) ────────────────────────────────
    crossover_rate: float = 0.9
    eta_c: float = 10.0                 # Indice de distribution SBX
                                        # faible → exploratoire, élevé → local

    # ── Mutation - Polynomiale (Deb & Goyal, 1996) ────────────────────────────
    mutation_rate: Optional[float] = None  # None → 1/n_genes (recommandé)
    eta_m: float = 20.0                 # Indice de distribution PM
    eta_m_min: float = 2.0
    eta_m_max: float = 200.0

    # ── Adaptation σ - Règle 1/5 (Rechenberg, 1973) ───────────────────────────
    adaptive_mutation: bool = True
    adaptation_window: int = 30         # Fenêtre d'évaluation du taux de succès
    adaptation_factor: float = 0.85     # Facteur c de Rechenberg ∈ (0, 1)

    # ── Élitisme & Hall of Fame ────────────────────────────────────────────────
    elite_count: int = 2                # Individus garantis dans la génération suivante
    n_hall_of_fame: int = 10            # Archive des N meilleurs all-time

    # ── Diversité & redémarrage ────────────────────────────────────────────────
    diversity_threshold: float = 0.02   # Diversité normalisée minimale ∈ [0, 1]
    stagnation_window: int = 20         # Générations sans amélioration significative
    stagnation_eps: float = 1e-8        # Seuil d'amélioration minimale
    restart_fraction: float = 0.40      # Fraction de la pop remplacée au redémarrage

    # ── Parallélisme & cache ───────────────────────────────────────────────────
    n_workers: int = 0
    use_cache: bool = True
    cache_decimals: int = 4             # Précision d'arrondi pour le cache

    # ── Critère d'arrêt anticipé ──────────────────────────────────────────────
    fitness_threshold: Optional[float] = None  # Arrêt si best ≤ seuil
    time_budget: Optional[float] = None        # Budget en secondes (None = illimité)

    def __post_init__(self):
        if self.mutation_rate is None:
            # Taux standard : 1/n (un gène muté en moyenne par individu)
            self.mutation_rate = 1.0 / self.n_genes
        if self.selection not in ("tournament", "rank"):
            raise ValueError(
                f"selection doit être 'tournament' ou 'rank', reçu '{self.selection}'"
            )


# ==============================================================================
# Individu
# ==============================================================================

class Individual:
    """Génome + fitness + génération de naissance."""

    __slots__ = ("genome", "fitness", "generation")

    def __init__(self, genome: np.ndarray, fitness: float = np.inf,
                 generation: int = 0):
        self.genome = genome.copy()
        self.fitness = fitness
        self.generation = generation

    def copy(self) -> "Individual":
        return Individual(self.genome, self.fitness, self.generation)

    def __repr__(self) -> str:
        return f"Individual(fitness={self.fitness:.6e}, gen={self.generation})"

    def __lt__(self, other: "Individual") -> bool:
        """Ordre naturel : fitness croissante (minimisation)."""
        return self.fitness < other.fitness


# ==============================================================================
# Latin Hypercube Sampling - McKay et al. (1979)
# ==============================================================================

def latin_hypercube_sampling(n_samples: int, n_dims: int,
                              low: float, high: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Génère n_samples points par Latin Hypercube Sampling (LHS).

    Chaque dimension est partitionnée en n_samples strates équiprobables et
    exactement un point est tiré par strate, ce qui garantit une couverture
    de l'espace de recherche bien supérieure à un échantillonnage aléatoire
    uniforme (notamment pour des problèmes à grande dimension).

    Référence : McKay, Beckman & Conover (1979), Technometrics, 21(2), 239-245.
    """
    cut = np.linspace(0.0, 1.0, n_samples + 1)
    u = np.empty((n_samples, n_dims))
    for j in range(n_dims):
        perm = rng.permutation(n_samples)
        u[:, j] = cut[perm] + rng.uniform(0.0, 1.0 / n_samples, n_samples)
    return low + u * (high - low)


# ==============================================================================
# Croisement SBX - Deb & Agrawal (1995)
# ==============================================================================

def sbx_crossover(p1: np.ndarray, p2: np.ndarray,
                  eta_c: float, low: float, high: float,
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Simulated Binary Crossover (SBX) - version bornée.

    Mimique le croisement mono-point binaire dans l'espace continu. La
    probabilité de générer un enfant entre les parents est identique à celle
    d'un croisement binaire avec le même η_c.

    L'indice η_c contrôle l'étalement de la distribution :
        η_c → 0   : enfants dispersés dans [low, high]  (exploration)
        η_c → ∞   : enfants proches des parents          (exploitation)

    Chaque gène est croisé indépendamment avec probabilité 0.5.

    Référence : Deb & Agrawal (1995), Complex Systems, 9(2), 115-148.
                Deb (2001), Multi-Objective Optimization, Wiley, eq. 9.9.
    """
    n = len(p1)
    c1, c2 = p1.copy(), p2.copy()

    for i in range(n):
        if rng.random() > 0.5:          # croisement génique indépendant
            continue
        if abs(p1[i] - p2[i]) < 1e-14:  # parents identiques → pas de croisement
            continue

        x1 = min(p1[i], p2[i])
        x2 = max(p1[i], p2[i])

        # ── Enfant vers la borne inférieure ───────────────────────────────────
        beta = 1.0 + 2.0 * (x1 - low) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        u = rng.random()
        beta_q = ((u * alpha) ** (1.0 / (eta_c + 1.0)) if u <= 1.0 / alpha
                  else (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0)))
        o1 = np.clip(0.5 * ((x1 + x2) - beta_q * (x2 - x1)), low, high)

        # ── Enfant vers la borne supérieure ───────────────────────────────────
        beta = 1.0 + 2.0 * (high - x2) / (x2 - x1)
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        u = rng.random()
        beta_q = ((u * alpha) ** (1.0 / (eta_c + 1.0)) if u <= 1.0 / alpha
                  else (1.0 / (2.0 - u * alpha)) ** (1.0 / (eta_c + 1.0)))
        o2 = np.clip(0.5 * ((x1 + x2) + beta_q * (x2 - x1)), low, high)

        # Affectation aléatoire pour éviter le biais directionnel
        if rng.random() > 0.5:
            o1, o2 = o2, o1

        c1[i], c2[i] = o1, o2

    return c1, c2


# ==============================================================================
# Mutation Polynomiale - Deb & Goyal (1996)
# ==============================================================================

def polynomial_mutation(x: np.ndarray, eta_m: float,
                         low: float, high: float, p_m: float,
                         rng: np.random.Generator) -> np.ndarray:
    """Mutation polynomiale bornée.

    Génère une perturbation Δ ∈ [-(x-low), (high-x)] suivant une distribution
    polynomiale bornée. L'indice η_m contrôle la concentration :
        η_m → 0   : perturbations larges (exploration)
        η_m → ∞   : perturbations infinitésimales (exploitation)

    Le résultat x' = x + Δ·(high-low) vérifie x' ∈ [low, high] par
    construction (pas de clamp nécessaire, mais appliqué par sécurité).

    Référence : Deb & Goyal (1996), Computer Science and Informatics, 26(4).
                Deb (2001), Multi-Objective Optimization, Wiley, eq. 9.14.
    """
    y = x.copy()
    delta_max = high - low

    for i in range(len(x)):
        if rng.random() >= p_m:
            continue
        u = rng.random()
        if u < 0.5:
            delta_L = (x[i] - low) / delta_max
            delta = ((2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta_L) ** (eta_m + 1.0))
                     ** (1.0 / (eta_m + 1.0)) - 1.0)
        else:
            delta_R = (high - x[i]) / delta_max
            delta = (1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5)
                            * (1.0 - delta_R) ** (eta_m + 1.0))
                     ** (1.0 / (eta_m + 1.0)))
        y[i] = np.clip(x[i] + delta * delta_max, low, high)

    return y


# ==============================================================================
# Opérateurs de sélection
# ==============================================================================

def tournament_select(population: List[Individual], k: int,
                      rng: np.random.Generator) -> Individual:
    """Sélection par tournoi stochastique de taille k.

    Tire k candidats au hasard et retourne le meilleur (fitness minimale).
    Complexité O(k), pression sélective contrôlée par k.
    """
    idx = rng.integers(0, len(population), size=k)
    return min((population[i] for i in idx), key=lambda ind: ind.fitness)


def rank_select(population: List[Individual], pressure: float,
                rng: np.random.Generator) -> Individual:
    """Sélection par rang linéaire (Baker, 1985).

    La population doit être triée par fitness croissante (population[0] = meilleur).

    Probabilité de sélection au rang r (r=0 = meilleur) parmi N individus :
        P(r) = (2-s)/N + 2(s-1)(N-1-r) / (N(N-1))
    où s ∈ [1.0, 2.0] est la pression sélective.

    s=1.0 → sélection uniforme, s=2.0 → pression maximale.

    Référence : Baker (1985), Proc. 1st ICGA, 101-111.
    """
    n = len(population)
    s = pressure
    # Probabilité décroissante avec le rang (meilleur = rang 0 = probabilité max)
    probs = np.array(
        [(2.0 - s) / n + 2.0 * (s - 1.0) * (n - 1 - r) / (n * (n - 1))
         for r in range(n)],
        dtype=np.float64
    )
    probs = np.maximum(probs, 0.0)
    probs /= probs.sum()
    return population[rng.choice(n, p=probs)]


# ==============================================================================
# Métrique de diversité de la population
# ==============================================================================

def population_diversity(population: List[Individual],
                          low: float, high: float) -> float:
    """Diversité normalisée de la population.

    Calcule l'écart-type moyen de chaque gène sur l'ensemble de la population,
    normalisé par le demi-étendue (high-low)/2.

    Retourne 0.0 (population convergée) à ~1.0 (maximalement dispersée).
    """
    if len(population) < 2:
        return 0.0
    genomes = np.array([ind.genome for ind in population])
    gene_std = np.std(genomes, axis=0)
    half_range = (high - low) / 2.0
    return float(np.mean(gene_std) / half_range)


# ==============================================================================
# Cache LRU pour les évaluations FDTD coûteuses
# ==============================================================================

class FitnessCache:
    """Cache à éviction LRU basé sur le hachage MD5 du génome arrondi.

    L'arrondi à `decimals` décimales traite comme identiques des génomes
    numériquement proches, économisant des évaluations FDTD redondantes.
    """

    def __init__(self, fitness_fn: Callable, decimals: int = 4,
                 maxsize: int = 8192):
        self.fitness_fn = fitness_fn
        self.decimals = decimals
        self.maxsize = maxsize
        self._cache: dict = {}
        self._order: list = []
        self.hits = 0
        self.misses = 0

    def _key(self, genome: np.ndarray) -> bytes:
        return hashlib.md5(np.round(genome, self.decimals).tobytes()).digest()

    def __call__(self, genome: np.ndarray) -> float:
        key = self._key(genome)
        if key in self._cache:
            self.hits += 1
            return self._cache[key]
        self.misses += 1
        value = self.fitness_fn(genome)
        if len(self._order) >= self.maxsize:
            del self._cache[self._order.pop(0)]
        self._cache[key] = value
        self._order.append(key)
        return value

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ==============================================================================
# Évaluation parallèle (multiprocessing)
# ==============================================================================

_GLOBAL_FITNESS_FN = None


def _mp_init(fn: Callable):
    global _GLOBAL_FITNESS_FN
    _GLOBAL_FITNESS_FN = fn


def _mp_eval(genome: np.ndarray) -> float:
    return _GLOBAL_FITNESS_FN(genome)


class ParallelEvaluator:
    """Évaluateur parallèle via multiprocessing.Pool.

    Note : CuPy (GPU) est incompatible avec le fork multiprocessing
    (cudaErrorInitializationError). Quand GPU_AVAILABLE est True, on
    bascule automatiquement en évaluation séquentielle - le GPU parallélise
    déjà les opérations matricielles internes.
    """

    def __init__(self, fitness_fn: Callable, n_workers: int):
        from src.utils.xp import GPU_AVAILABLE
        self.fitness_fn = fitness_fn
        # Force sequential when GPU is active (CUDA context can't be forked)
        self.n_workers = 1 if GPU_AVAILABLE else n_workers
        self._pool: Optional[mp.Pool] = None

    def evaluate(self, genomes: List[np.ndarray]) -> List[float]:
        if self.n_workers <= 1 or len(genomes) <= 1:
            return [self.fitness_fn(g) for g in genomes]
        if self._pool is None:
            self._pool = mp.Pool(
                self.n_workers,
                initializer=_mp_init,
                initargs=(self.fitness_fn,)
            )
        return self._pool.map(_mp_eval, genomes)

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


# ==============================================================================
# Hall of Fame - archive des N meilleurs individus all-time
# ==============================================================================

class HallOfFame:
    """Maintient les N meilleurs individus jamais évalués (tri croissant)."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self._members: List[Individual] = []

    def update(self, population: List[Individual]):
        for ind in population:
            if np.isinf(ind.fitness):
                continue
            if len(self._members) < self.max_size:
                self._members.append(ind.copy())
                self._members.sort()
            elif ind.fitness < self._members[-1].fitness:
                self._members[-1] = ind.copy()
                self._members.sort()

    @property
    def best(self) -> Optional[Individual]:
        return self._members[0] if self._members else None

    def __iter__(self):
        return iter(self._members)

    def __len__(self):
        return len(self._members)


# ==============================================================================
# Adaptation σ - Règle 1/5 de Rechenberg (1973)
# ==============================================================================

class OneFifthRule:
    """Adapte l'indice η_m de la mutation polynomiale via la règle du 1/5.

    Principe (Rechenberg, 1973) : le taux de succès optimal des mutations
    est 1/5. Sur une fenêtre glissante de W mutations :
        - taux > 1/5  →  mutations trop conservatives  →  diminuer η_m (σ↑)
        - taux < 1/5  →  mutations trop agressives     →  augmenter η_m (σ↓)

    Pour la mutation polynomiale, η_m élevé ↔ perturbations faibles (σ petit).
    L'adaptation applique le facteur c de Rechenberg (classiquement c = 0.85).

    Référence : Rechenberg (1973), Evolutionstrategie. Frommann-Holzboog.
    """

    TARGET: float = 0.2  # 1/5

    def __init__(self, eta_m_init: float, eta_m_min: float, eta_m_max: float,
                 window: int, factor: float):
        self.eta_m = eta_m_init
        self.eta_m_min = eta_m_min
        self.eta_m_max = eta_m_max
        self.window = window
        self.c = factor
        self._outcomes: List[int] = []    # 1 = succès, 0 = échec

    def record(self, parent_fitness: float, child_fitness: float):
        """Enregistre si la mutation a amélioré l'individu parent."""
        self._outcomes.append(int(child_fitness < parent_fitness))
        if len(self._outcomes) > self.window:
            self._outcomes.pop(0)

    def adapt(self):
        """Ajuste η_m si suffisamment de données sont disponibles."""
        if len(self._outcomes) < max(5, self.window // 4):
            return
        rate = float(np.mean(self._outcomes))
        if rate > self.TARGET:
            # Trop de succès → explorer davantage → réduire η_m
            self.eta_m = max(self.eta_m_min, self.eta_m * self.c)
        else:
            # Peu de succès → affiner → augmenter η_m
            self.eta_m = min(self.eta_m_max, self.eta_m / self.c)

    @property
    def success_rate(self) -> float:
        return float(np.mean(self._outcomes)) if self._outcomes else float("nan")


# ==============================================================================
# Algorithme Génétique
# ==============================================================================

class GeneticAlgorithm:
    """Algorithme Génétique scientifique pour l'optimisation continue.

    Pipeline par génération
    -----------------------
    1. Tri de la population + mise à jour du Hall of Fame
    2. Enregistrement des statistiques (fitness, diversité, adaptation)
    3. Élitisme : conservation des `elite_count` meilleurs individus
    4. Génération des enfants :
         sélection (tournoi ou rang) → SBX → mutation polynomiale
    5. Évaluation (parallèle si n_workers > 1, avec cache)
    6. Adaptation de η_m via la règle 1/5 de Rechenberg
    7. Détection stagnation / collapse de diversité → redémarrage LHS
    8. Vérification des critères d'arrêt
    """

    def __init__(self, config: GAConfig,
                 fitness_fn: Callable[[np.ndarray], float]):
        self.cfg = config
        self._raw_fitness = fitness_fn
        self.rng = np.random.default_rng(config.seed)

        # Cache optionnel
        self._eval_fn: Callable = (
            FitnessCache(fitness_fn, decimals=config.cache_decimals)
            if config.use_cache else fitness_fn
        )

        # Évaluateur (parallèle ou séquentiel)
        self._evaluator = ParallelEvaluator(self._eval_fn, config.n_workers)

        # Adaptation 1/5 de Rechenberg
        self._adapt: Optional[OneFifthRule] = (
            OneFifthRule(config.eta_m, config.eta_m_min, config.eta_m_max,
                         config.adaptation_window, config.adaptation_factor)
            if config.adaptive_mutation else None
        )

        # Hall of Fame
        self._hof = HallOfFame(config.n_hall_of_fame)

        # État courant
        self.population: List[Individual] = []
        self.generation: int = 0
        self._n_evals: int = 0
        self._n_restarts: int = 0

        # Historique complet (pour visualisation et diagnostic)
        self.history: dict = {
            "best_fitness":  [],   # Meilleure fitness all-time (HOF)
            "mean_fitness":  [],   # Fitness moyenne de la population
            "worst_fitness": [],   # Pire fitness de la population
            "std_fitness":   [],   # Écart-type des fitness
            "diversity":     [],   # Diversité normalisée [0, 1]
            "eta_m":         [],   # η_m courant (adaptatif ou fixe)
            "success_rate":  [],   # Taux de succès des mutations
            "n_evals":       [],   # Nombre cumulé d'évaluations FDTD
            "restarts":      [],   # Nombre de redémarrages déclenchés
            "best_genome":   [],   # Meilleur génome all-time (HOF)
        }

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation (LHS)
    # ──────────────────────────────────────────────────────────────────────────

    def _initialize(self):
        cfg = self.cfg
        lhs = latin_hypercube_sampling(
            cfg.pop_size, cfg.n_genes, cfg.gene_min, cfg.gene_max, self.rng
        )
        self.population = [
            Individual(lhs[i], generation=0) for i in range(cfg.pop_size)
        ]
        self._evaluate_pending()

    # ──────────────────────────────────────────────────────────────────────────
    # Évaluation
    # ──────────────────────────────────────────────────────────────────────────

    def _evaluate_pending(self):
        """Évalue uniquement les individus avec fitness = ∞."""
        pending = [ind for ind in self.population if np.isinf(ind.fitness)]
        if not pending:
            return
        fitnesses = self._evaluator.evaluate([ind.genome for ind in pending])
        for ind, f in zip(pending, fitnesses):
            ind.fitness = f
        self._n_evals += len(pending)

    # ──────────────────────────────────────────────────────────────────────────
    # Sélection
    # ──────────────────────────────────────────────────────────────────────────

    def _select(self) -> Individual:
        cfg = self.cfg
        if cfg.selection == "tournament":
            return tournament_select(self.population, cfg.tournament_size,
                                     self.rng)
        return rank_select(self.population, cfg.rank_pressure, self.rng)

    # ──────────────────────────────────────────────────────────────────────────
    # Une génération complète
    # ──────────────────────────────────────────────────────────────────────────

    def _step(self):
        cfg = self.cfg
        self.population.sort()

        n_elite = max(1, cfg.elite_count)
        new_pop: List[Individual] = [ind.copy() for ind in self.population[:n_elite]]

        eta_m = self._adapt.eta_m if self._adapt else cfg.eta_m
        mutation_records: List[Tuple[float, Individual]] = []   # (parent_f, child)

        while len(new_pop) < cfg.pop_size:
            p1 = self._select()
            p2 = self._select()

            if self.rng.random() < cfg.crossover_rate:
                g1, g2 = sbx_crossover(
                    p1.genome, p2.genome,
                    cfg.eta_c, cfg.gene_min, cfg.gene_max, self.rng
                )
            else:
                g1, g2 = p1.genome.copy(), p2.genome.copy()

            g1 = polynomial_mutation(g1, eta_m, cfg.gene_min, cfg.gene_max,
                                      cfg.mutation_rate, self.rng)
            g2 = polynomial_mutation(g2, eta_m, cfg.gene_min, cfg.gene_max,
                                      cfg.mutation_rate, self.rng)

            c1 = Individual(g1, generation=self.generation + 1)
            c2 = Individual(g2, generation=self.generation + 1)

            mutation_records.append((p1.fitness, c1))
            new_pop.append(c1)
            if len(new_pop) < cfg.pop_size:
                mutation_records.append((p2.fitness, c2))
                new_pop.append(c2)

        self.population = new_pop[:cfg.pop_size]
        self._evaluate_pending()

        # Adaptation 1/5 après évaluation
        if self._adapt is not None:
            for parent_f, child in mutation_records:
                if not np.isinf(child.fitness):
                    self._adapt.record(parent_f, child.fitness)
            self._adapt.adapt()

        self.generation += 1

    # ──────────────────────────────────────────────────────────────────────────
    # Détection stagnation / redémarrage adaptatif
    # ──────────────────────────────────────────────────────────────────────────

    def _check_restart(self) -> bool:
        """Remplace une fraction de la population si stagnation ou collapse."""
        cfg = self.cfg
        h = self.history

        div_collapse = (h["diversity"][-1] < cfg.diversity_threshold
                        if h["diversity"] else False)

        stagnated = False
        w = cfg.stagnation_window
        if len(h["best_fitness"]) >= w:
            window_best = h["best_fitness"][-w:]
            stagnated = (window_best[0] - window_best[-1]) < cfg.stagnation_eps

        if not (div_collapse or stagnated):
            return False

        # Garder les meilleurs, remplacer le reste par LHS
        self.population.sort()
        n_keep = max(1, int(cfg.pop_size * (1.0 - cfg.restart_fraction)))
        n_new = cfg.pop_size - n_keep

        new_genomes = latin_hypercube_sampling(
            n_new, cfg.n_genes, cfg.gene_min, cfg.gene_max, self.rng
        )
        new_inds = [Individual(new_genomes[i], generation=self.generation)
                    for i in range(n_new)]
        fitnesses = self._evaluator.evaluate([ind.genome for ind in new_inds])
        for ind, f in zip(new_inds, fitnesses):
            ind.fitness = f
        self._n_evals += n_new

        self.population = self.population[:n_keep] + new_inds
        self._n_restarts += 1
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Enregistrement des statistiques
    # ──────────────────────────────────────────────────────────────────────────

    def _record(self):
        fitnesses = [ind.fitness for ind in self.population
                     if not np.isinf(ind.fitness)]
        best_f = self._hof.best.fitness if self._hof.best else np.inf
        eta_m = self._adapt.eta_m if self._adapt else self.cfg.eta_m
        sr = self._adapt.success_rate if self._adapt else float("nan")

        self.history["best_fitness"].append(best_f)
        self.history["mean_fitness"].append(
            float(np.mean(fitnesses)) if fitnesses else np.inf
        )
        self.history["worst_fitness"].append(
            float(np.max(fitnesses)) if fitnesses else np.inf
        )
        self.history["std_fitness"].append(
            float(np.std(fitnesses)) if fitnesses else 0.0
        )
        self.history["diversity"].append(
            population_diversity(self.population, self.cfg.gene_min,
                                 self.cfg.gene_max)
        )
        self.history["eta_m"].append(eta_m)
        self.history["success_rate"].append(sr)
        self.history["n_evals"].append(self._n_evals)
        self.history["restarts"].append(self._n_restarts)
        self.history["best_genome"].append(
            self._hof.best.genome.copy() if self._hof.best else None
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Boucle principale
    # ──────────────────────────────────────────────────────────────────────────

    def run(self, verbose: bool = True,
            checkpoint_fn=None) -> Individual:
        """Lance l'optimisation et retourne le meilleur individu all-time.

        Parameters
        ----------
        verbose :
            Affiche la progression en temps réel (Rich).
        checkpoint_fn :
            Callable(gen, best_individual) appelé après chaque génération.
            Permet de sauvegarder des checkpoints externes.
        """
        from src.utils.console import GADisplay

        cfg = self.cfg
        t0 = time.perf_counter()

        display = GADisplay(cfg) if verbose else None
        if display:
            display.start()

        # Initialisation par LHS
        self._initialize()
        self.population.sort()
        self._hof.update(self.population)
        self._record()
        if display:
            display.update(0, self.history, 0.0, False)

        for gen in range(cfg.n_generations):
            t_gen = time.perf_counter()
            self._step()
            self.population.sort()
            self._hof.update(self.population)

            restarted = self._check_restart()
            if restarted:
                self.population.sort()
                self._hof.update(self.population)

            self._record()

            if display:
                display.update(gen + 1, self.history,
                               time.perf_counter() - t_gen, restarted)

            if checkpoint_fn is not None and self._hof.best is not None:
                checkpoint_fn(gen + 1, self._hof.best)

            # Arrêt anticipé - seuil de fitness
            if (cfg.fitness_threshold is not None
                    and self._hof.best is not None
                    and self._hof.best.fitness <= cfg.fitness_threshold):
                if display:
                    display._progress.console.print(
                        f"[green]  Convergence atteinte (fitness ≤ "
                        f"{cfg.fitness_threshold:.2e})[/green]"
                    )
                break

            # Arrêt anticipé - budget temps
            if (cfg.time_budget is not None
                    and time.perf_counter() - t0 >= cfg.time_budget):
                if display:
                    mins = cfg.time_budget / 60
                    display._progress.console.print(
                        f"[yellow]  Budget temps atteint ({mins:.0f} min)[/yellow]"
                    )
                break

        self._evaluator.close()
        elapsed = time.perf_counter() - t0

        if display:
            display.finish(elapsed, self)

        return self._hof.best

