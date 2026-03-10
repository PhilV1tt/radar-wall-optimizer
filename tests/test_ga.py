"""Tests pour l'algorithme génétique et CMA-ES."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from src.optim import (
    GeneticAlgorithm, GAConfig, Individual,
    CMAES, CMAConfig, FitnessCache
)


def sphere_function(x: np.ndarray) -> float:
    """Fonction de test simple : f(x) = sum(x²). Minimum à x=0."""
    return float(np.sum(x**2))


def rastrigin(x: np.ndarray) -> float:
    """Fonction de Rastrigin : test multimodal."""
    n = len(x)
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


class TestIndividual:
    """Tests de la classe Individual."""

    def test_copy_genome(self):
        g = np.array([1.0, 2.0, 3.0])
        ind = Individual(g)
        g[0] = 999
        assert ind.genome[0] == 1.0

    def test_default_fitness(self):
        ind = Individual(np.zeros(5))
        assert ind.fitness == np.inf


class TestGA:
    """Tests de l'algorithme génétique."""

    def test_convergence_sphere(self):
        cfg = GAConfig(n_genes=5, pop_size=15, n_generations=30,
                      eta_m=20.0, use_cache=False)
        ga = GeneticAlgorithm(cfg, sphere_function)
        best = ga.run(verbose=False)
        assert best.fitness < 0.5, f"GA n'a pas convergé : f={best.fitness}"

    def test_history_tracking(self):
        n_gen = 5
        cfg = GAConfig(n_genes=3, pop_size=8, n_generations=n_gen, use_cache=False)
        ga = GeneticAlgorithm(cfg, sphere_function)
        ga.run(verbose=False)
        # n_gen + 1 entrées : génération 0 (init) + n_gen générations
        assert len(ga.history['best_fitness']) == n_gen + 1
        assert len(ga.history['best_genome']) == n_gen + 1

    def test_fitness_decreasing(self):
        cfg = GAConfig(n_genes=3, pop_size=10, n_generations=10, use_cache=False)
        ga = GeneticAlgorithm(cfg, sphere_function)
        ga.run(verbose=False)
        bests = ga.history['best_fitness']
        for i in range(1, len(bests)):
            assert bests[i] <= bests[i-1], "Élitisme violé"

    def test_population_size(self):
        cfg = GAConfig(n_genes=3, pop_size=12, n_generations=3, use_cache=False)
        ga = GeneticAlgorithm(cfg, sphere_function)
        ga.run(verbose=False)
        assert len(ga.population) == 12

    def test_gene_bounds(self):
        cfg = GAConfig(n_genes=5, pop_size=10, n_generations=10,
                      gene_min=-1.0, gene_max=1.0, use_cache=False)
        ga = GeneticAlgorithm(cfg, sphere_function)
        ga.run(verbose=False)
        for ind in ga.population:
            assert np.all(ind.genome >= -1.0)
            assert np.all(ind.genome <= 1.0)


class TestCMAES:
    """Tests du CMA-ES."""

    def test_convergence_sphere(self):
        cfg = CMAConfig(n_params=5, sigma0=0.5, max_iter=50)
        cma = CMAES(cfg, sphere_function)
        best = cma.run(verbose=False)
        assert cma.best_fitness < 0.1, f"CMA-ES n'a pas convergé : f={cma.best_fitness}"

    def test_warm_start(self):
        cfg = CMAConfig(n_params=3, sigma0=0.3, max_iter=20)
        cma = CMAES(cfg, sphere_function)
        x0 = np.array([0.1, 0.1, 0.1])
        best = cma.run(x0=x0, verbose=False)
        assert cma.best_fitness < 0.05

    def test_param_bounds(self):
        cfg = CMAConfig(n_params=5, sigma0=0.5, max_iter=20,
                       param_min=-1.0, param_max=1.0)
        cma = CMAES(cfg, sphere_function)
        cma.run(verbose=False)
        assert np.all(cma.best_params >= -1.0)
        assert np.all(cma.best_params <= 1.0)


class TestFitnessCache:
    """Tests du cache de fitness."""

    def test_cache_hit(self):
        call_count = 0
        def counting_fn(x):
            nonlocal call_count
            call_count += 1
            return float(np.sum(x**2))

        cache = FitnessCache(counting_fn, decimals=3)
        x = np.array([0.1234, 0.5678])

        r1 = cache(x)
        r2 = cache(x)

        assert r1 == r2
        assert cache.hits == 1
        assert cache.misses == 1
        assert call_count == 1

    def test_cache_similar_genomes(self):
        cache = FitnessCache(sphere_function, decimals=3)
        x1 = np.array([0.1234, 0.5678])
        x2 = np.array([0.12345, 0.56784])

        cache(x1)
        cache(x2)
        assert cache.hits == 1
