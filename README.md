# Radar Wall Optimizer

Optimisation de géométrie de mur anti-radar par simulation électromagnétique FDTD 2D et algorithmes d'apprentissage automatique.

## Aperçu

Ce projet cherche la forme optimale d'un mur pour **minimiser la section efficace radar (RCS)** — c'est-à-dire la quantité d'énergie électromagnétique rétrodiffusée vers la source radar. La géométrie optimale "piège" les ondes en les absorbant ou en les diffusant dans des directions non dangereuses.

**Applications :** matériaux absorbant les ondes radar (RAM), technologie furtive, conception de structures anti-radar.

---

## Comment ça marche

```
Onde plane incidente → Simulation FDTD 2D → Calcul RCS → Optimisation → Mur optimal
```

1. **Simulation FDTD** : résout les équations de Maxwell en polarisation TMz (Ez, Hx, Hy) sur une grille de Yee 2D
2. **Fonction fitness** : énergie rétrodiffusée mesurée via transformation Proche-Lointain Champ (NTFF)
3. **Optimisation** : trois algorithmes comparés pour trouver le profil de mur optimal
4. **Visualisation** : figures de convergence, profils de mur, cartes de champ

---

## Démarrage rapide

```bash
git clone https://github.com/PhilV1tt/radar-wall-optimizer.git
cd radar-wall-optimizer
pip install -r requirements.txt

# Lancer une optimisation rapide (~3 min)
python run.py fast

# Pipeline complet avec visualisations (~20 min)
python run.py medium
```

---

## CLI — `run.py`

Point d'entrée unifié avec quatre presets :

```
python run.py <preset> [options]
```

| Preset | Durée | Description |
|---|---|---|
| `fast` | ~3 min | GA rapide, petite grille — feedback de développement |
| `medium` | ~20 min | GA complet + graphiques de convergence et de champ |
| `full` | ~40 min | GA + CMA-ES + RL + comparaison complète |
| `validate` | ~5 min | Tests de convergence physique FDTD |

**Options :**

```
--workers N      Nombre de workers parallèles (défaut : cpu // 2)
--seed N         Graine aléatoire pour la reproductibilité
--out DIR        Dossier de sortie (défaut : results/YYYYMMDD_HHMMSS/)
--no-plots       Ne pas générer les graphiques matplotlib
--checkpoint N   Sauvegarder un checkpoint tous les N générations (défaut : 25)
--time MINUTES   Budget temps en minutes — arrête le GA proprement à la fin
                 de la génération courante (compatible fast et medium)
```

**Exemples :**

```bash
# Lancement standard
python run.py fast

# 8 workers, graine fixe, dossier personnalisé
python run.py medium --workers 8 --seed 42 --out results/mon_experience

# GA pendant exactement 40 min (arrêt propre, résultats sauvegardés)
python run.py medium --time 40

# GA rapide pendant 10 min
python run.py fast --time 10

# Sans graphiques (serveur headless)
python run.py medium --no-plots

# Tests de validation FDTD
python run.py validate
```

Chaque run crée automatiquement un sous-dossier horodaté dans `results/` avec :
- `best_<preset>.npz` — meilleur génome + historique de convergence
- `checkpoint_genNNNN.npz` — checkpoints intermédiaires (reprise en cas de crash)
- `convergence_<preset>.png` — courbe de convergence (si `--no-plots` non activé)
- `field_<preset>.png` — champ électrique optimal + géométrie du mur

---

## Affichage terminal

L'optimisation affiche en temps réel un panel de configuration, une table de générations et une barre de progression avec ETA :

```
╭────────────── GA · Optimisation mur anti-radar ──────────────╮
│ Population            48 individus · 100 générations         │
│ Gènes                 16 · bornes [-1.0, 1.0]                │
│ Croisement            SBX η_c=10 · p_c=0.90                  │
│ Mutation              PM adaptatif 1/5 · η_m=20 · p_m=0.0625 │
│ Sélection             tournoi k=3                            │
│ Évaluation            8 workers                              │
│ Backend               NumPy (CPU)                            │
╰──────────────────────────────────────────────────────────────╯
    Gen           Best           Mean          Std      Div    η_m   Succ  Info
────────────────────────────────────────────────────────────────────────────────
      0   2.134e-01   2.589e-01   4.256e-02   1.0000   20.0     —   init
      1   1.987e-01   2.412e-01   3.892e-02   0.9876   17.0   21%   9.2s
      2   1.654e-01   2.198e-01   3.102e-02   0.9432   14.5   28%   9.1s
 ─ 3/100 ━━━━━━━━━━━━━ 3% 0:00:27 · 0:14:13 · 1.65e-01 ─
```

---

## Accélération GPU

Le projet détecte automatiquement le backend de calcul au démarrage :

- **GPU NVIDIA (CUDA)** → CuPy (accélération GPU)
- **CPU** (macOS Apple Silicon, CI, pas de CUDA) → NumPy

Pour activer CuPy sur une machine NVIDIA, installer selon la version CUDA :

```bash
pip install cupy-cuda12x   # CUDA 12.x
pip install cupy-cuda11x   # CUDA 11.x
```

Aucune modification du code n'est nécessaire — la détection est automatique.

---

## Architecture du projet

```
radar-wall-optimizer/
├── run.py                  # Point d'entrée unifié (CLI)
├── src/
│   ├── fdtd/               # Simulation électromagnétique
│   │   ├── config.py       # Paramètres physiques et configuration FDTD
│   │   ├── core.py         # Classe principale FDTD2D_TMz
│   │   ├── materials.py    # Géométrie du mur PEC
│   │   ├── tfsf.py         # Frontière Total-Field / Scattered-Field
│   │   ├── pml.py          # Couche PML absorbante (CPML)
│   │   └── ntff.py         # Calcul RCS via Near-to-Far-Field
│   ├── optim/              # Algorithmes d'optimisation
│   │   ├── genetic.py      # Algorithme génétique (SBX + mutation polynomiale)
│   │   ├── cmaes.py        # CMA-ES (Covariance Matrix Adaptation)
│   │   ├── rl_agent.py     # Apprentissage par renforcement (REINFORCE)
│   │   └── fitness.py      # Wrapper d'évaluation FDTD
│   ├── utils/
│   │   ├── xp.py           # Détection GPU (CuPy / NumPy)
│   │   └── console.py      # Affichage Rich (progress bar, panels)
│   └── viz/                # Visualisation
│       ├── plots.py        # Figures scientifiques
│       ├── animation.py    # Animation temps réel
│       └── dashboard.py    # Dashboard interactif
├── scripts/                # Scripts spécialisés (longues durées)
│   ├── run_optimization.py # Pipeline complet (GA → CMA-ES → RL)
│   ├── run_overnight.py    # Sweep nuit (5000 générations)
│   ├── run_validation.py   # Tests de convergence physique
│   └── live_demo.py        # Visualisation interactive
├── tests/
│   ├── test_fdtd.py
│   ├── test_ga.py
│   └── test_ntff.py
├── results/                # Sorties générées (ignoré par git)
└── requirements.txt
```

---

## Installation

**Prérequis :** Python 3.10+

```bash
pip install -r requirements.txt

# Optionnel
pip install cma>=3.3        # CMA-ES via bibliothèque officielle
pip install scipy>=1.11     # Interpolation avancée
pip install pytest          # Tests unitaires
```

---

## Algorithmes d'optimisation

| Algorithme | Forces | Paramètres clés |
|---|---|---|
| **Génétique (GA)** | Exploration globale, robuste aux minima locaux | `pop_size`, `n_generations`, SBX + mutation polynomiale |
| **CMA-ES** | Convergence fine, adapte covariance et pas | Warm-start depuis le meilleur GA |
| **RL (REINFORCE)** | Capture les corrélations entre segments | `n_episodes`, décroissance de σ |

Le GA utilise :
- **Latin Hypercube Sampling (LHS)** pour l'initialisation — couverture uniforme de l'espace
- **SBX** (Simulated Binary Crossover) — croisement en continu
- **Mutation polynomiale** avec critère **1/5 de Rechenberg** — adaptation automatique
- **Redémarrage adaptatif** — détection de stagnation et collapse de diversité

---

## Configuration FDTD

```python
FDTDConfig(
    nx=150, ny=150,     # Grille 150×150 cellules
    ppw=15,             # 15 points par longueur d'onde
    freq=10e9,          # 10 GHz (bande X)
    courant=0.5,        # Nombre de Courant (stabilité CFL)
    n_steps=350,        # Pas temporels
)
```

La simulation utilise :
- **Grille de Yee** décalée (Ez aux nœuds entiers, Hx/Hy aux demi-nœuds)
- **CPML** (Convolutional PML) pour absorber les ondes sortantes
- **TFSF** pour injecter l'onde plane incidente à angle variable
- **Mur PEC** paramétrique à N points de contrôle

---

## Références scientifiques

- **Yee, K.S.** (1966) — Algorithme FDTD original
- **Deb & Agrawal** (1995) — Simulated Binary Crossover (SBX)
- **Deb & Goyal** (1996) — Mutation polynomiale bornée
- **Rechenberg, I.** (1973) — Critère 1/5 pour l'adaptation du pas
- **McKay et al.** (1979) — Latin Hypercube Sampling
- **Baker, J.E.** (1985) — Sélection par rang linéaire

---

## Stack technique

- **Python 3.10+**
- **NumPy ≥ 1.24** — calculs numériques (CPU)
- **Matplotlib ≥ 3.7** — visualisation scientifique
- **Rich ≥ 13.0** — affichage terminal
- **CuPy** *(optionnel)* — accélération GPU NVIDIA
- **CMA ≥ 3.3** *(optionnel)* — CMA-ES
- **SciPy ≥ 1.11** *(optionnel)* — interpolation avancée
