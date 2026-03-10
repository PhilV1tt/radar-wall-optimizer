"""
Apprentissage par Renforcement — REINFORCE avec baseline.
"""

import numpy as np
from typing import Callable, List, Dict, Tuple
from dataclasses import dataclass
import time


@dataclass
class RLConfig:
    """Configuration de l'agent RL."""
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
    """Politique paramétrique simple (réseau linéaire + sortie gaussienne)."""

    def __init__(self, n_params: int, lr: float):
        self.n_params = n_params
        self.lr = lr
        self.W = np.eye(n_params) * 0.1 + np.random.randn(n_params, n_params) * 0.01
        self.b = np.zeros(n_params)

    def forward(self, state: np.ndarray) -> np.ndarray:
        return self.W @ state + self.b

    def sample_action(self, state: np.ndarray, std: float) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.forward(state)
        noise = np.random.randn(self.n_params) * std
        action = mean + noise

        log_prob = -0.5 * np.sum((action - mean)**2) / (std**2) \
                   - 0.5 * self.n_params * np.log(2 * np.pi * std**2)

        return action, log_prob

    def log_probability(self, state: np.ndarray, action: np.ndarray, std: float) -> float:
        mean = self.forward(state)
        return -0.5 * np.sum((action - mean)**2) / (std**2) \
               - 0.5 * self.n_params * np.log(2 * np.pi * std**2)

    def update(self, states: List[np.ndarray], actions: List[np.ndarray],
               advantages: List[float], std: float):
        grad_W = np.zeros_like(self.W)
        grad_b = np.zeros_like(self.b)

        for s, a, adv in zip(states, actions, advantages):
            mean = self.forward(s)
            diff = (a - mean) / (std**2)
            grad_W += np.outer(diff, s) * adv
            grad_b += diff * adv

        n = len(states)
        if n > 0:
            grad_W /= n
            grad_b /= n

            grad_norm = np.sqrt(np.sum(grad_W**2) + np.sum(grad_b**2))
            max_norm = 1.0
            if grad_norm > max_norm:
                grad_W *= max_norm / grad_norm
                grad_b *= max_norm / grad_norm

            self.W += self.lr * grad_W
            self.b += self.lr * grad_b


class RLOptimizer:
    """Optimiseur par apprentissage par renforcement."""

    def __init__(self, config: RLConfig, fitness_fn: Callable[[np.ndarray], float]):
        self.cfg = config
        self.fitness_fn = fitness_fn
        self.policy = SimplePolicy(config.n_params, config.learning_rate)

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

            action_scale = 0.2
            new_state = np.clip(state + action_scale * action, -1.0, 1.0)

            rcs = self.fitness_fn(new_state)
            reward = -rcs

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
        gamma = self.cfg.gamma
        returns = []
        G = 0.0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        return returns

    def run(self, verbose: bool = True) -> np.ndarray:
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

            for _ in range(cfg.n_rollouts):
                if np.random.random() < 0.3 and self.best_fitness < np.inf:
                    init = self.best_genome + np.random.randn(cfg.n_params) * 0.1
                    init = np.clip(init, -1, 1)
                else:
                    init = np.random.uniform(-1, 1, cfg.n_params)

                traj = self.run_episode(std, init)
                returns = self.compute_returns(traj['rewards'])
                advantages = [G - self.baseline for G in returns]

                all_states.extend(traj['states'])
                all_actions.extend(traj['actions'])
                all_advantages.extend(advantages)
                episode_rewards.append(sum(traj['rewards']))

                mean_return = np.mean(returns)
                self.baseline = (cfg.baseline_momentum * self.baseline +
                                (1 - cfg.baseline_momentum) * mean_return)

            self.policy.update(all_states, all_actions, all_advantages, std)
            std = max(cfg.action_std_min, std * cfg.std_decay)

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
