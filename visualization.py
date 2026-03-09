"""
================================================================================
Module de Visualisation — Résultats FDTD et Optimisation
================================================================================

Génère toutes les figures scientifiques du projet :
1. Champ Ez à différents instants (snapshots)
2. Géométrie du mur (profil PEC)
3. Convergence de l'optimisation (GA et RL)
4. Comparaison des profils optimaux
5. Diagramme polaire de RCS bistatique
6. Space-time diagram du champ
================================================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Backend non-interactif
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec
from typing import Optional, List, Dict, Tuple
import os


def setup_style():
    """Configure le style scientifique des figures."""
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 150,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
    })

setup_style()


def plot_field_snapshot(Ez: np.ndarray, pec_mask: np.ndarray,
                        title: str = "Champ Ez", 
                        save_path: Optional[str] = None,
                        vmax: Optional[float] = None):
    """Trace un snapshot du champ Ez avec le mur PEC.
    
    Le champ est affiché avec une colormap divergente (bleu-blanc-rouge)
    centrée sur zéro, superposé au masque PEC en noir.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    if vmax is None:
        vmax = max(abs(Ez.max()), abs(Ez.min()), 1e-10)
    
    # Champ Ez
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.imshow(Ez.T, origin='lower', cmap='RdBu_r', norm=norm,
                    aspect='equal', interpolation='bilinear')
    
    # PEC en noir semi-transparent
    pec_display = np.ma.masked_where(~pec_mask.T, np.ones_like(pec_mask.T, dtype=float))
    ax.imshow(pec_display, origin='lower', cmap='Greys', alpha=0.8,
              aspect='equal', vmin=0, vmax=1)
    
    ax.set_xlabel('x (cellules)')
    ax.set_ylabel('y (cellules)')
    ax.set_title(title)
    
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, label='Ez (V/m)')
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_field_snapshots_grid(snapshots: List[Tuple[np.ndarray, np.ndarray, str]],
                               save_path: Optional[str] = None):
    """Trace une grille de snapshots (pour montrer l'évolution temporelle)."""
    n = len(snapshots)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    vmax = max(max(abs(s[0].max()), abs(s[0].min()), 1e-10) for s in snapshots)
    
    for idx, (Ez, pec, title) in enumerate(snapshots):
        ax = axes[idx]
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax.imshow(Ez.T, origin='lower', cmap='RdBu_r', norm=norm,
                        aspect='equal', interpolation='bilinear')
        
        pec_display = np.ma.masked_where(~pec.T, np.ones_like(pec.T, dtype=float))
        ax.imshow(pec_display, origin='lower', cmap='Greys', alpha=0.8,
                  aspect='equal', vmin=0, vmax=1)
        
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        
    # Cacher les axes inutilisés
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)
        
    fig.suptitle('Évolution temporelle du champ Ez', fontsize=14, y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_wall_profile(params: np.ndarray, title: str = "Profil du mur",
                       save_path: Optional[str] = None):
    """Trace le profil de surface du mur à partir des paramètres."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    
    n = len(params)
    y = np.linspace(0, 1, n)
    
    # Surface du mur
    ax.fill_betweenx(y, params, params + 0.15, color='gray', alpha=0.7, label='Mur PEC')
    ax.plot(params, y, 'k-', linewidth=2, label='Surface exposée')
    
    # Points de contrôle
    ax.plot(params, y, 'ro', markersize=5, zorder=5)
    
    ax.set_xlabel('Déplacement normalisé')
    ax.set_ylabel('Position le long du mur (normalisée)')
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(-1.3, 1.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='Mur plat')
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_ga_convergence(history: Dict, save_path: Optional[str] = None):
    """Trace les courbes de convergence de l'algorithme génétique."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    gens = range(1, len(history['best_fitness']) + 1)
    
    # Fitness
    ax1.semilogy(gens, history['best_fitness'], 'b-', linewidth=2, label='Meilleure')
    ax1.semilogy(gens, history['mean_fitness'], 'g--', linewidth=1, label='Moyenne')
    ax1.fill_between(gens, history['best_fitness'], history['worst_fitness'],
                      alpha=0.1, color='green')
    ax1.set_xlabel('Génération')
    ax1.set_ylabel('RCS (σ/λ) — log')
    ax1.set_title('Convergence de l\'Algorithme Génétique')
    ax1.legend()
    
    # Profil du meilleur individu au fil des générations
    genomes = np.array(history['best_genome'])
    im = ax2.imshow(genomes.T, aspect='auto', cmap='coolwarm',
                     extent=[1, len(gens), 0, genomes.shape[1]],
                     interpolation='nearest')
    ax2.set_xlabel('Génération')
    ax2.set_ylabel('Segment du mur')
    ax2.set_title('Évolution du profil optimal')
    plt.colorbar(im, ax=ax2, label='Déplacement')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_rl_convergence(history: Dict, save_path: Optional[str] = None):
    """Trace les courbes de convergence du RL."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    eps = range(1, len(history['best_fitness']) + 1)
    
    ax1.semilogy(eps, history['best_fitness'], 'r-', linewidth=2, label='Meilleure RCS')
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('RCS (σ/λ) — log')
    ax1.set_title('Convergence du Reinforcement Learning')
    ax1.legend()
    
    ax2.plot(eps, history['action_std'], 'purple', linewidth=2)
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('σ (action)')
    ax2.set_title('Décroissance de l\'écart-type d\'exploration')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_comparison(ga_best_params: np.ndarray, ga_best_fitness: float,
                     rl_best_params: np.ndarray, rl_best_fitness: float,
                     flat_fitness: float,
                     save_path: Optional[str] = None):
    """Compare les profils optimaux trouvés par GA et RL."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    
    n = len(ga_best_params)
    y = np.linspace(0, 1, n)
    
    # Mur plat (référence)
    axes[0].fill_betweenx(y, 0, 0.15, color='gray', alpha=0.7)
    axes[0].plot(np.zeros(n), y, 'k-', linewidth=2)
    axes[0].set_title(f'Mur plat (baseline)\nRCS = {flat_fitness:.4f} σ/λ')
    axes[0].set_xlim(-1.3, 1.3)
    axes[0].set_xlabel('Déplacement')
    axes[0].set_ylabel('Position')
    
    # GA optimal
    axes[1].fill_betweenx(y, ga_best_params, ga_best_params + 0.15, 
                           color='steelblue', alpha=0.7)
    axes[1].plot(ga_best_params, y, 'b-', linewidth=2)
    axes[1].set_title(f'Algorithme Génétique\nRCS = {ga_best_fitness:.4f} σ/λ')
    axes[1].set_xlim(-1.3, 1.3)
    axes[1].set_xlabel('Déplacement')
    
    # RL optimal
    axes[2].fill_betweenx(y, rl_best_params, rl_best_params + 0.15,
                           color='indianred', alpha=0.7)
    axes[2].plot(rl_best_params, y, 'r-', linewidth=2)
    axes[2].set_title(f'Reinforcement Learning\nRCS = {rl_best_fitness:.4f} σ/λ')
    axes[2].set_xlim(-1.3, 1.3)
    axes[2].set_xlabel('Déplacement')
    
    fig.suptitle('Comparaison des profils de mur optimaux', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_rcs_polar(angles: np.ndarray, rcs_flat: np.ndarray, 
                    rcs_ga: np.ndarray, rcs_rl: np.ndarray,
                    save_path: Optional[str] = None):
    """Trace le diagramme polaire de RCS bistatique."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), subplot_kw={'projection': 'polar'})
    
    # RCS en dB
    rcs_flat_db = 10 * np.log10(np.maximum(rcs_flat, 1e-10))
    rcs_ga_db = 10 * np.log10(np.maximum(rcs_ga, 1e-10))
    rcs_rl_db = 10 * np.log10(np.maximum(rcs_rl, 1e-10))
    
    ax.plot(angles, rcs_flat_db, 'k-', linewidth=1.5, label='Mur plat', alpha=0.7)
    ax.plot(angles, rcs_ga_db, 'b-', linewidth=2, label='GA optimal')
    ax.plot(angles, rcs_rl_db, 'r--', linewidth=2, label='RL optimal')
    
    # Marquer la direction de rétrodiffusion (φ = π)
    ax.axvline(x=np.pi, color='green', linestyle=':', alpha=0.5)
    ax.annotate('Rétrodiffusion', xy=(np.pi, ax.get_ylim()[1]),
                fontsize=9, ha='center', color='green')
    
    ax.set_title('Section Efficace Radar Bistatique\nσ/λ (dB)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig


def plot_project_summary(sim_config, ga_history, rl_history,
                          ga_best, rl_best, flat_fitness,
                          save_path: Optional[str] = None):
    """Crée une figure de synthèse du projet complet."""
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    n = len(ga_best)
    y = np.linspace(0, 1, n)
    
    # --- Panel 1 : Schéma du problème ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    
    # Radar
    ax1.annotate('RADAR', xy=(1, 5), fontsize=12, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='round', facecolor='gold', alpha=0.5))
    
    # Onde
    for i in range(3):
        x = np.linspace(2, 7, 100)
        wave = 5 + 0.3 * np.sin(2*np.pi*x - i*0.5) * np.exp(-0.1*(x-4)**2)
        ax1.plot(x, wave, 'b-', alpha=0.3)
    ax1.annotate('→ Onde incidente', xy=(4, 6), fontsize=9, color='blue')
    
    # Mur
    ax1.fill_between([7, 7.5], 2, 8, color='gray', alpha=0.7)
    ax1.annotate('MUR', xy=(7.25, 5), fontsize=10, fontweight='bold',
                 ha='center', va='center', rotation=90)
    
    ax1.set_title('Schéma du problème', fontsize=11)
    ax1.axis('off')
    
    # --- Panel 2 : Convergence GA ---
    ax2 = fig.add_subplot(gs[0, 1])
    gens = range(1, len(ga_history['best_fitness']) + 1)
    ax2.semilogy(gens, ga_history['best_fitness'], 'b-', linewidth=2, label='GA Best')
    ax2.semilogy(gens, ga_history['mean_fitness'], 'b--', linewidth=1, alpha=0.5, label='GA Mean')
    if rl_history:
        eps = range(1, len(rl_history['best_fitness']) + 1)
        ax2.semilogy(eps, rl_history['best_fitness'], 'r-', linewidth=2, label='RL Best')
    ax2.axhline(y=flat_fitness, color='gray', linestyle=':', label='Mur plat')
    ax2.set_xlabel('Itération')
    ax2.set_ylabel('RCS (σ/λ)')
    ax2.set_title('Convergence')
    ax2.legend(fontsize=8)
    
    # --- Panel 3 : Profils ---
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(np.zeros(n), y, 'k--', linewidth=1, label='Plat')
    ax3.fill_betweenx(y, ga_best, ga_best + 0.1, color='steelblue', alpha=0.5)
    ax3.plot(ga_best, y, 'b-', linewidth=2, label='GA')
    ax3.fill_betweenx(y, rl_best, rl_best + 0.1, color='indianred', alpha=0.5)
    ax3.plot(rl_best, y, 'r-', linewidth=2, label='RL')
    ax3.set_xlabel('Déplacement')
    ax3.set_ylabel('Position')
    ax3.set_title('Profils optimaux')
    ax3.legend(fontsize=8)
    ax3.set_xlim(-1.3, 1.3)
    
    # --- Panel 4-6 : Texte résumé ---
    ax4 = fig.add_subplot(gs[1, :])
    ax4.axis('off')
    
    summary_text = (
        f"╔══════════════════════════════════════════════════════════════════╗\n"
        f"║  RÉSUMÉ DU PROJET : Optimisation de mur anti-radar par ML      ║\n"
        f"╠══════════════════════════════════════════════════════════════════╣\n"
        f"║  Simulation : FDTD 2D TMz ({sim_config.nx}×{sim_config.ny}, "
        f"f={sim_config.freq/1e9:.0f} GHz, λ={sim_config.wavelength*100:.1f} cm)  ║\n"
        f"║  Résolution : {sim_config.ppw} points/λ, "
        f"Δx={sim_config.dx*1e3:.2f} mm, Sc={sim_config.courant}               ║\n"
        f"║                                                                ║\n"
        f"║  Mur plat (baseline) : RCS = {flat_fitness:.6f} σ/λ              ║\n"
        f"║  GA optimal          : RCS = {ga_history['best_fitness'][-1]:.6f} σ/λ  "
        f"(réduction: {(1-ga_history['best_fitness'][-1]/flat_fitness)*100:.1f}%)  ║\n"
        f"║  RL optimal          : RCS = {rl_history['best_fitness'][-1]:.6f} σ/λ  "
        f"(réduction: {(1-rl_history['best_fitness'][-1]/flat_fitness)*100:.1f}%)  ║\n"
        f"╚══════════════════════════════════════════════════════════════════╝"
    )
    
    ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes,
             fontsize=10, fontfamily='monospace', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.suptitle('Optimisation de Géométrie de Mur Anti-Radar\nFDTD 2D + Machine Learning',
                 fontsize=15, fontweight='bold')
    
    if save_path:
        fig.savefig(save_path)
        plt.close(fig)
    return fig
