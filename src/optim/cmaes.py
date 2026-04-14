"""
CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
"""

import numpy as np
from typing import Callable, List, Optional
from dataclasses import dataclass
import time

from src.optim.genetic import ParallelEvaluator


@dataclass
class CMAConfig:
    """Configuration du CMA-ES."""
    n_params: int = 20
    sigma0: float = 0.5
    max_iter: int = 100
    pop_size: int = 0       # 0 = auto (4 + 3*ln(n))
    param_min: float = -1.0
    param_max: float = 1.0
    n_workers: int = 0


class CMAES:
    """CMA-ES - adapte la matrice de covariance et le pas σ automatiquement."""

    def __init__(self, config: CMAConfig, fitness_fn: Callable[[np.ndarray], float]):
        self.cfg = config
        self.fitness_fn = fitness_fn
        self.n = config.n_params

        lam = config.pop_size if config.pop_size > 0 else 4 + int(3 * np.log(self.n))
        self.lam = lam
        self.mu = lam // 2

        raw_weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = raw_weights / raw_weights.sum()
        self.mu_eff = 1.0 / np.sum(self.weights ** 2)

        n = self.n
        self.c_sigma = (self.mu_eff + 2) / (n + self.mu_eff + 5)
        self.d_sigma = 1.0 + 2.0 * max(0, np.sqrt((self.mu_eff - 1) / (n + 1)) - 1) + self.c_sigma
        self.c_c = (4 + self.mu_eff / n) / (n + 4 + 2 * self.mu_eff / n)
        self.c1 = 2.0 / ((n + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c1, 2 * (self.mu_eff - 2 + 1 / self.mu_eff) / ((n + 2) ** 2 + self.mu_eff))

        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        self.mean = np.zeros(n)
        self.sigma = config.sigma0
        self.C = np.eye(n)
        self.p_sigma = np.zeros(n)
        self.p_c = np.zeros(n)

        self.best_fitness = np.inf
        self.best_params = np.zeros(n)
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
            'best_genome': [],
            'sigma': [],
        }

        self.parallel = ParallelEvaluator(
            fitness_fn, config.n_workers
        ) if config.n_workers > 1 else None

    def _sample_population(self) -> List[np.ndarray]:
        try:
            sqrtC = np.linalg.cholesky(self.C)
        except np.linalg.LinAlgError:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            self.C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            sqrtC = np.linalg.cholesky(self.C)

        population = []
        for _ in range(self.lam):
            z = np.random.randn(self.n)
            x = self.mean + self.sigma * (sqrtC @ z)
            x = np.clip(x, self.cfg.param_min, self.cfg.param_max)
            population.append(x)
        return population

    def _update(self, population: List[np.ndarray], fitnesses: np.ndarray):
        n = self.n
        indices = np.argsort(fitnesses)
        sorted_pop = [population[i] for i in indices]

        old_mean = self.mean.copy()

        self.mean = np.zeros(n)
        for i in range(self.mu):
            self.mean += self.weights[i] * sorted_pop[i]

        try:
            eigvals, eigvecs = np.linalg.eigh(self.C)
            eigvals = np.maximum(eigvals, 1e-10)
            inv_sqrtC = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        except np.linalg.LinAlgError:
            inv_sqrtC = np.eye(n)

        self.p_sigma = ((1 - self.c_sigma) * self.p_sigma
                        + np.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
                        * inv_sqrtC @ (self.mean - old_mean) / self.sigma)

        h_sigma = (np.linalg.norm(self.p_sigma)
                   / np.sqrt(1 - (1 - self.c_sigma) ** (2 * (len(self.history['best_fitness']) + 1)))
                   < (1.4 + 2 / (n + 1)) * self.chi_n)

        self.p_c = ((1 - self.c_c) * self.p_c
                    + h_sigma * np.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)
                    * (self.mean - old_mean) / self.sigma)

        rank1 = np.outer(self.p_c, self.p_c)

        rank_mu = np.zeros((n, n))
        for i in range(self.mu):
            yi = (sorted_pop[i] - old_mean) / self.sigma
            rank_mu += self.weights[i] * np.outer(yi, yi)

        self.C = ((1 - self.c1 - self.c_mu) * self.C
                  + self.c1 * rank1
                  + self.c_mu * rank_mu)

        self.sigma *= np.exp(
            (self.c_sigma / self.d_sigma)
            * (np.linalg.norm(self.p_sigma) / self.chi_n - 1)
        )
        self.sigma = min(self.sigma, 2.0)

    def run(self, x0: Optional[np.ndarray] = None, verbose: bool = True) -> np.ndarray:
        cfg = self.cfg
        self.mean = x0.copy() if x0 is not None else np.zeros(self.n)

        if verbose:
            print("=" * 70)
            print("CMA-ES - Optimisation de géométrie de mur anti-radar")
            print("=" * 70)
            print(f"Dimension: {self.n} | λ={self.lam} | μ={self.mu} | σ₀={cfg.sigma0}")
            mode = f"multiprocessing ({cfg.n_workers} workers)" if cfg.n_workers > 1 else "séquentiel"
            print(f"Mode: {mode}")
            print("-" * 70)

        t_start = time.time()

        for iteration in range(cfg.max_iter):
            t_iter = time.time()

            population = self._sample_population()

            if self.parallel is not None:
                fitnesses = np.array(self.parallel.evaluate_batch(population))
            else:
                fitnesses = np.array([self.fitness_fn(x) for x in population])

            best_idx = np.argmin(fitnesses)
            if fitnesses[best_idx] < self.best_fitness:
                self.best_fitness = fitnesses[best_idx]
                self.best_params = population[best_idx].copy()

            self.history['best_fitness'].append(self.best_fitness)
            self.history['mean_fitness'].append(np.mean(fitnesses))
            self.history['worst_fitness'].append(np.max(fitnesses))
            self.history['best_genome'].append(self.best_params.copy())
            self.history['sigma'].append(self.sigma)

            self._update(population, fitnesses)

            dt = time.time() - t_iter

            if verbose and (iteration + 1) % 5 == 0:
                print(f"Iter {iteration+1:3d}/{cfg.max_iter} | "
                      f"Best: {self.best_fitness:.6f} | Mean: {np.mean(fitnesses):.6f} | "
                      f"σ: {self.sigma:.4f} | Time: {dt:.1f}s")

            if self.sigma < 1e-8:
                if verbose:
                    print(f"Convergence atteinte (σ = {self.sigma:.2e})")
                break

        if self.parallel is not None:
            self.parallel.close()

        total_time = time.time() - t_start
        if verbose:
            print("-" * 70)
            print(f"Optimisation terminée en {total_time:.1f}s")
            print(f"Meilleure fitness: {self.best_fitness:.6f}")
            print("=" * 70)

        return self.best_params
