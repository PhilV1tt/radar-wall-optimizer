
import numpy as np
from typing import Callable, List, Tuple, Optional
from dataclasses import dataclass
import time


@dataclass
class GAConfig:
    n_genes: int = 20
    pop_size: int = 100
    n_generations: int = 500
    mutation_rate: float = 0.15
    mutation_sigma: float = 0.3
    crossover_rate: float = 0.8
    elite_fraction: float = 0.1
    gene_min: float = -1.0
    gene_max: float = 1.0
    tournament_size: int = 3


class Individual:
    
    def __init__(self, genome: np.ndarray, fitness: float = np.inf):
        self.genome = genome.copy()
        self.fitness = fitness
        
    def __repr__(self):
        return f"Individual(fitness={self.fitness:.6f})"


class GeneticAlgorithm:
    
    def __init__(self, config: GAConfig, fitness_fn: Callable[[np.ndarray], float]):
        self.cfg = config
        self.fitness_fn = fitness_fn
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history = {
            'best_fitness': [],
            'mean_fitness': [],
            'worst_fitness': [],
            'best_genome': [],
        }
        
    def initialize_population(self):
        """Initialise la population avec des profils aléatoires.
        
        On inclut aussi un mur plat (genome = 0) comme référence,
        et quelques profils "dentelés" inspirés de la technologie stealth.
        """
        cfg = self.cfg
        self.population = []
        
        # 1. Mur plat (baseline)
        flat = np.zeros(cfg.n_genes)
        self.population.append(Individual(flat))
        
        # 2. Profils dentelés (inspiration stealth)
        sawtooth = np.zeros(cfg.n_genes)
        for i in range(cfg.n_genes):
            sawtooth[i] = (i % 4 - 1.5) / 1.5  # dents de scie
        self.population.append(Individual(sawtooth))
        
        # 3. Profil en V
        v_profile = np.abs(np.linspace(-1, 1, cfg.n_genes)) * 2 - 1
        self.population.append(Individual(v_profile))
        
        # 4. Le reste : aléatoire
        for _ in range(cfg.pop_size - 3):
            genome = np.random.uniform(cfg.gene_min, cfg.gene_max, cfg.n_genes)
            self.population.append(Individual(genome))
            
    def evaluate_population(self):
        """Évalue la fitness de tous les individus non évalués."""
        for ind in self.population:
            if ind.fitness == np.inf:  # pas encore évalué
                ind.fitness = self.fitness_fn(ind.genome)
                
    def tournament_selection(self) -> Individual:
        """Sélection par tournoi.
        
        On tire aléatoirement tournament_size individus et on garde le meilleur.
        C'est une méthode de sélection à pression ajustable.
        """
        contestants = np.random.choice(len(self.population), 
                                        self.cfg.tournament_size, 
                                        replace=False)
        best_idx = min(contestants, key=lambda i: self.population[i].fitness)
        return self.population[best_idx]
    
    def blx_alpha_crossover(self, parent1: Individual, parent2: Individual,
                             alpha: float = 0.5) -> Tuple[Individual, Individual]:
        """Croisement BLX-α (Blend Crossover).
        
        Pour chaque gène i, l'intervalle [min(p1_i, p2_i), max(p1_i, p2_i)]
        est étendu de α de chaque côté, et l'enfant est tiré uniformément
        dans cet intervalle étendu.
        
        Avantage : explore au-delà de l'enveloppe convexe des parents,
        ce qui favorise l'exploration de l'espace de recherche.
        """
        cfg = self.cfg
        g1 = parent1.genome
        g2 = parent2.genome
        
        d = np.abs(g1 - g2)
        low = np.minimum(g1, g2) - alpha * d
        high = np.maximum(g1, g2) + alpha * d
        
        child1_genome = np.random.uniform(low, high)
        child2_genome = np.random.uniform(low, high)
        
        # Clipping aux bornes
        child1_genome = np.clip(child1_genome, cfg.gene_min, cfg.gene_max)
        child2_genome = np.clip(child2_genome, cfg.gene_min, cfg.gene_max)
        
        return Individual(child1_genome), Individual(child2_genome)
    
    def mutate(self, individual: Individual, generation: int) -> Individual:
        """Mutation gaussienne avec adaptation du pas.
        
        Le pas de mutation diminue au fil des générations (cooling schedule) :
        σ(g) = σ₀ × (1 - g/G)^0.5
        
        Ceci favorise l'exploration au début et l'exploitation à la fin.
        """
        cfg = self.cfg
        genome = individual.genome.copy()
        
        # Adaptation du pas
        progress = generation / max(1, cfg.n_generations)
        sigma = cfg.mutation_sigma * (1.0 - progress) ** 0.5
        
        for i in range(len(genome)):
            if np.random.random() < cfg.mutation_rate:
                genome[i] += np.random.normal(0, sigma)
                genome[i] = np.clip(genome[i], cfg.gene_min, cfg.gene_max)
                
        return Individual(genome)
    
    def evolve_one_generation(self, generation: int):
        """Fait évoluer la population d'une génération.
        
        Étapes :
        1. Tri par fitness
        2. Élitisme : copie des meilleurs
        3. Création de la nouvelle génération par sélection + croisement + mutation
        4. Évaluation des nouveaux individus
        """
        cfg = self.cfg
        
        # Tri par fitness (minimisation)
        self.population.sort(key=lambda ind: ind.fitness)
        
        # Élitisme
        n_elite = max(1, int(cfg.pop_size * cfg.elite_fraction))
        new_population = [Individual(ind.genome, ind.fitness) 
                         for ind in self.population[:n_elite]]
        
        # Remplir le reste par croisement + mutation
        while len(new_population) < cfg.pop_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            
            if np.random.random() < cfg.crossover_rate:
                child1, child2 = self.blx_alpha_crossover(parent1, parent2)
            else:
                child1 = Individual(parent1.genome.copy())
                child2 = Individual(parent2.genome.copy())
                
            child1 = self.mutate(child1, generation)
            child2 = self.mutate(child2, generation)
            
            new_population.append(child1)
            if len(new_population) < cfg.pop_size:
                new_population.append(child2)
                
        self.population = new_population[:cfg.pop_size]
        
        # Évaluer les nouveaux individus
        self.evaluate_population()
        
        # Mettre à jour le meilleur
        self.population.sort(key=lambda ind: ind.fitness)
        current_best = self.population[0]
        if self.best_individual is None or current_best.fitness < self.best_individual.fitness:
            self.best_individual = Individual(current_best.genome, current_best.fitness)
            
        # Historique
        fitnesses = [ind.fitness for ind in self.population]
        self.history['best_fitness'].append(min(fitnesses))
        self.history['mean_fitness'].append(np.mean(fitnesses))
        self.history['worst_fitness'].append(max(fitnesses))
        self.history['best_genome'].append(self.best_individual.genome.copy())
        
    def run(self, verbose: bool = True) -> Individual:
        """Lance l'optimisation génétique complète.
        
        Returns
        -------
        best : Individual
            Le meilleur individu trouvé (mur optimal).
        """
        cfg = self.cfg
        
        if verbose:
            print("=" * 70)
            print("ALGORITHME GÉNÉTIQUE — Optimisation de géométrie de mur anti-radar")
            print("=" * 70)
            print(f"Population: {cfg.pop_size} | Générations: {cfg.n_generations}")
            print(f"Gènes: {cfg.n_genes} | Mutation: σ={cfg.mutation_sigma}, p={cfg.mutation_rate}")
            print("-" * 70)
        
        # Initialisation
        t_start = time.time()
        self.initialize_population()
        self.evaluate_population()
        
        # Meilleur initial
        self.population.sort(key=lambda ind: ind.fitness)
        self.best_individual = Individual(
            self.population[0].genome, self.population[0].fitness
        )
        
        for gen in range(cfg.n_generations):
            t_gen = time.time()
            self.evolve_one_generation(gen)
            dt = time.time() - t_gen
            
            if verbose:
                best_f = self.history['best_fitness'][-1]
                mean_f = self.history['mean_fitness'][-1]
                print(f"Gen {gen+1:3d}/{cfg.n_generations} | "
                      f"Best: {best_f:.6f} | Mean: {mean_f:.6f} | "
                      f"Time: {dt:.1f}s")
                
        total_time = time.time() - t_start
        if verbose:
            print("-" * 70)
            print(f"Optimisation terminée en {total_time:.1f}s")
            print(f"Meilleure fitness: {self.best_individual.fitness:.6f}")
            print("=" * 70)
            
        return self.best_individual
