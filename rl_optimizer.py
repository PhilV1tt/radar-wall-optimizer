"""
================================================================================
Module Apprentissage par Renforcement — Optimisation de géométrie de mur
================================================================================

Implémente une approche RL simplifiée (policy gradient / REINFORCE avec baseline)
pour l'optimisation de la géométrie du mur anti-radar.

Avertissement scientifique
--------------------------
Le RL n'est PAS l'outil optimal pour ce problème car :
- Il n'y a pas de séquence de décisions (le profil est statique)
- Il n'y a pas d'interaction temporelle avec un environnement
- Le "state" ne change pas au fil de l'épisode

On reformule le problème en un MDP dégénéré :
- State s : état courant du profil (vecteur de paramètres)
- Action a : modification du profil (delta sur chaque paramètre)  
- Reward r : -RCS (on veut minimiser la RCS, donc maximiser -RCS)
- Transition : s' = clip(s + a, -1, 1)
- Épisode : N étapes de modification séquentielle du profil

L'avantage par rapport au GA : le RL peut apprendre une POLITIQUE de 
modification qui capture des corrélations entre segments voisins.

Implémentation : REINFORCE avec baseline (Monte Carlo Policy Gradient)
======================================================================

La politique π_θ(a|s) est un réseau de neurones qui, étant donné l'état
courant du profil, produit une distribution gaussienne sur les actions.

L'objectif est de maximiser :
    J(θ) = E_τ~π_θ [Σ_t γ^t r_t]

Le gradient de politique (théorème REINFORCE) :
    ∇_θ J = E [Σ_t ∇_θ log π_θ(a_t|s_t) · (R_t - b)]

où b est une baseline (valeur moyenne des returns) qui réduit la variance.

Référence : Williams (1992), Sutton & Barto (2018)
================================================================================
"""

import numpy as np
from typing import Callable, List, Dict, Tuple
from dataclasses import dataclass
import time


@dataclass
class RLConfig:
    """Configuration de l'agent RL.
    
    Paramètres
    ----------
    n_params : int
        Dimension de l'espace d'action (= nombre de segments du mur).
    n_episodes : int
        Nombre d'épisodes d'entraînement.
    steps_per_episode : int
        Nombre d'étapes de modification par épisode.
    learning_rate : float
        Taux d'apprentissage pour la mise à jour de la politique.
    gamma : float
        Facteur d'escompte.
    action_std_init : float
        Écart-type initial de la politique gaussienne.
    action_std_min : float
        Écart-type minimal (pour éviter l'effondrement).
    std_decay : float
        Facteur de décroissance de l'écart-type par épisode.
    baseline_momentum : float
        Momentum pour la baseline exponential moving average.
    n_rollouts : int
        Nombre de rollouts par épisode (pour réduire la variance).
    """
    n_params: int = 20
    n_episodes: int = 100
    steps_per_episode: int = 5
    learning_rate: float = 0.01
    gamma: float = 0.99
    action_std_init: float = 0.3
    action_std_min: float = 0.05
    std_decay: float = 0.995
    baseline_momentum: float = 0.9
    n_rollouts: int = 5


class SimplePolicy:
    """Politique paramétrique simple (réseau linéaire + sortie gaussienne).
    
    Architecture :
        État s ∈ R^n → Couche linéaire → μ ∈ R^n (moyenne de la gaussienne)
        
    L'action est échantillonnée : a ~ N(μ(s), σ²I)
    où σ est annealed au cours de l'entraînement.
    
    On utilise un réseau simple car :
    1. L'espace d'état et d'action ont la même dimension
    2. La relation état→action optimale est probablement proche de l'identité
    3. Un réseau complexe serait overfitting sur si peu de données
    """
    
    def __init__(self, n_params: int, lr: float):
        self.n_params = n_params
        self.lr = lr
        
        # Paramètres de la politique
        # W : matrice de poids (n_params × n_params)
        # b : biais (n_params)
        # Initialisation proche de l'identité (la politique initiale conserve le profil)
        self.W = np.eye(n_params) * 0.1 + np.random.randn(n_params, n_params) * 0.01
        self.b = np.zeros(n_params)
        
    def forward(self, state: np.ndarray) -> np.ndarray:
        """Calcule la moyenne de la politique μ(s) = W·s + b."""
        return self.W @ state + self.b
    
    def sample_action(self, state: np.ndarray, std: float) -> Tuple[np.ndarray, np.ndarray]:
        """Échantillonne une action de la politique.
        
        Returns
        -------
        action : np.ndarray
            Action échantillonnée a ~ N(μ(s), σ²I).
        log_prob : np.ndarray
            Log-probabilité de l'action sous la politique actuelle.
        """
        mean = self.forward(state)
        noise = np.random.randn(self.n_params) * std
        action = mean + noise
        
        # Log-probabilité gaussienne
        log_prob = -0.5 * np.sum((action - mean)**2) / (std**2) \
                   - 0.5 * self.n_params * np.log(2 * np.pi * std**2)
        
        return action, log_prob
    
    def log_probability(self, state: np.ndarray, action: np.ndarray, std: float) -> float:
        """Calcule log π_θ(a|s)."""
        mean = self.forward(state)
        return -0.5 * np.sum((action - mean)**2) / (std**2) \
               - 0.5 * self.n_params * np.log(2 * np.pi * std**2)
    
    def update(self, states: List[np.ndarray], actions: List[np.ndarray],
               advantages: List[float], std: float):
        """Met à jour la politique par gradient de politique (REINFORCE).
        
        ∇_θ log π_θ(a|s) pour une gaussienne N(Ws+b, σ²I) :
            ∂/∂W log π = (a - μ) × s^T / σ²
            ∂/∂b log π = (a - μ) / σ²
            
        Mise à jour : θ ← θ + α × ∇_θ log π × advantage
        """
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)
        
        for s, a, adv in zip(states, actions, advantages):
            mean = self.forward(s)
            diff = (a - mean) / (std**2)
            
            grad_W += np.outer(diff, s) * adv
            grad_b += diff * adv
            
        # Moyenne sur le batch
        n = len(states)
        if n > 0:
            grad_W /= n
            grad_b /= n
            
            # Clip du gradient pour stabilité
            grad_norm = np.sqrt(np.sum(grad_W**2) + np.sum(grad_b**2))
            max_norm = 1.0
            if grad_norm > max_norm:
                grad_W *= max_norm / grad_norm
                grad_b *= max_norm / grad_norm
            
            self.W += self.lr * grad_W
            self.b += self.lr * grad_b


class RLOptimizer:
    """Optimiseur par apprentissage par renforcement.
    
    Reformulation MDP :
    -------------------
    - State : profil courant du mur (vecteur ∈ [-1,1]^n)
    - Action : modification du profil (delta ∈ R^n)
    - Reward : -RCS_backscatter (on veut minimiser la RCS)
    - Transition : s_{t+1} = clip(s_t + α·a_t, -1, 1) où α contrôle l'amplitude
    - Terminal : après steps_per_episode étapes
    
    À chaque épisode, l'agent part d'un profil initial et le modifie
    séquentiellement. La récompense cumulative guide l'apprentissage.
    """
    
    def __init__(self, config: RLConfig, fitness_fn: Callable[[np.ndarray], float]):
        """
        Paramètres
        ----------
        config : RLConfig
        fitness_fn : callable
            Prend un génome et retourne la RCS (à minimiser).
        """
        self.cfg = config
        self.fitness_fn = fitness_fn
        self.policy = SimplePolicy(config.n_params, config.learning_rate)
        
        # Tracking
        self.best_genome = np.zeros(config.n_params)
        self.best_fitness = np.inf
        self.baseline = 0.0
        
        self.history = {
            'episode_reward': [],
            'best_fitness': [],
            'mean_reward': [],
            'action_std': [],
        }
        
    def run_episode(self, std: float, initial_state: np.ndarray = None) -> Dict:
        """Exécute un épisode complet.
        
        Returns
        -------
        trajectory : dict
            Contient states, actions, rewards, log_probs.
        """
        cfg = self.cfg
        
        if initial_state is None:
            initial_state = np.zeros(cfg.n_params)
            
        state = initial_state.copy()
        
        states = []
        actions = []
        rewards = []
        log_probs = []
        
        for step in range(cfg.steps_per_episode):
            action, log_prob = self.policy.sample_action(state, std)
            
            # Transition
            action_scale = 0.2  # amplitude de modification
            new_state = np.clip(state + action_scale * action, -1.0, 1.0)
            
            # Récompense = -RCS (on veut maximiser le reward = minimiser la RCS)
            rcs = self.fitness_fn(new_state)
            reward = -rcs
            
            # Tracker le meilleur
            if rcs < self.best_fitness:
                self.best_fitness = rcs
                self.best_genome = new_state.copy()
                
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = new_state
            
        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'log_probs': log_probs,
            'final_state': state,
        }
    
    def compute_returns(self, rewards: List[float]) -> List[float]:
        """Calcule les returns escomptés G_t = Σ_{k=0}^{T-t} γ^k r_{t+k}.
        
        Le return mesure la récompense cumulative future à partir de l'instant t.
        """
        gamma = self.cfg.gamma
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns
    
    def run(self, verbose: bool = True) -> np.ndarray:
        """Lance l'entraînement RL complet.
        
        Pour chaque épisode :
        1. Exécuter n_rollouts trajectoires
        2. Calculer les returns et avantages
        3. Mettre à jour la politique
        4. Décroître l'écart-type
        
        Returns
        -------
        best_genome : np.ndarray
            Le meilleur profil de mur trouvé.
        """
        cfg = self.cfg
        std = cfg.action_std_init
        
        if verbose:
            print("=" * 70)
            print("REINFORCEMENT LEARNING — Optimisation de géométrie de mur anti-radar")
            print("=" * 70)
            print(f"Épisodes: {cfg.n_episodes} | Steps/épisode: {cfg.steps_per_episode}")
            print(f"Rollouts: {cfg.n_rollouts} | LR: {cfg.learning_rate}")
            print("-" * 70)
        
        t_start = time.time()
        
        for episode in range(cfg.n_episodes):
            t_ep = time.time()
            
            all_states = []
            all_actions = []
            all_advantages = []
            episode_rewards = []
            
            # Exécuter plusieurs rollouts
            for _ in range(cfg.n_rollouts):
                # Départ : soit du meilleur connu, soit aléatoire
                if np.random.random() < 0.3 and self.best_fitness < np.inf:
                    init = self.best_genome + np.random.randn(cfg.n_params) * 0.1
                    init = np.clip(init, -1, 1)
                else:
                    init = np.random.uniform(-1, 1, cfg.n_params)
                    
                traj = self.run_episode(std, init)
                
                # Calculer les returns
                returns = self.compute_returns(traj['rewards'])
                
                # Avantages = returns - baseline
                advantages = [G - self.baseline for G in returns]
                
                all_states.extend(traj['states'])
                all_actions.extend(traj['actions'])
                all_advantages.extend(advantages)
                episode_rewards.append(sum(traj['rewards']))
                
                # Mise à jour de la baseline (moving average)
                mean_return = np.mean(returns)
                self.baseline = (cfg.baseline_momentum * self.baseline + 
                                (1 - cfg.baseline_momentum) * mean_return)
            
            # Mise à jour de la politique
            self.policy.update(all_states, all_actions, all_advantages, std)
            
            # Decay de l'écart-type
            std = max(cfg.action_std_min, std * cfg.std_decay)
            
            # Historique
            mean_rew = np.mean(episode_rewards)
            self.history['episode_reward'].append(mean_rew)
            self.history['best_fitness'].append(self.best_fitness)
            self.history['mean_reward'].append(mean_rew)
            self.history['action_std'].append(std)
            
            dt = time.time() - t_ep
            
            if verbose and (episode + 1) % 5 == 0:
                print(f"Ep {episode+1:3d}/{cfg.n_episodes} | "
                      f"Best RCS: {self.best_fitness:.6f} | "
                      f"Mean R: {mean_rew:.4f} | "
                      f"σ: {std:.4f} | Time: {dt:.1f}s")
                
        total = time.time() - t_start
        if verbose:
            print("-" * 70)
            print(f"Entraînement terminé en {total:.1f}s")
            print(f"Meilleure RCS: {self.best_fitness:.6f}")
            print("=" * 70)
            
        return self.best_genome
