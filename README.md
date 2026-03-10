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

## Architecture du projet

```
radar-wall-optimizer/
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
│   └── viz/                # Visualisation
│       ├── plots.py        # Figures scientifiques
│       ├── animation.py    # Animation temps réel
│       └── dashboard.py    # Dashboard interactif
├── scripts/
│   ├── run_optimization.py # Pipeline complet (GA → CMA-ES → RL → comparaison)
│   ├── run_20min.py        # Démo ~20 min (32 cœurs)
│   ├── run_ga_20min.py     # GA uniquement
│   ├── run_overnight.py    # Sweep complet (nuit)
│   ├── run_validation.py   # Tests de convergence physique
│   └── live_demo.py        # Visualisation interactive
├── tests/
│   ├── test_fdtd.py        # Tests simulation FDTD
│   ├── test_ga.py          # Tests algorithme génétique
│   └── test_ntff.py        # Tests calcul RCS
├── results/                # Figures générées automatiquement
├── requirements.txt
└── pyproject.toml
```

---

## Installation

**Prérequis :** Python 3.10+

```bash
# Cloner le dépôt
git clone https://github.com/PhilV1tt/radar-wall-optimizer.git
cd radar-wall-optimizer

# Installer les dépendances
pip install -r requirements.txt

# Optionnel : CMA-ES et SciPy
pip install cma>=3.3 scipy>=1.11

# Optionnel : tests
pip install pytest
```

---

## Utilisation

### Pipeline complet
```bash
python scripts/run_optimization.py
```
Lance la séquence complète : baseline → GA → CMA-ES → RL → figures comparatives.

### Démo rapide (~20 min)
```bash
python scripts/run_20min.py
```

### Algorithme génétique uniquement
```bash
python scripts/run_ga_20min.py
```

### Visualisation en temps réel
```bash
python scripts/live_demo.py
```

### Tests de validation physique
```bash
python scripts/run_validation.py
```

### Tests unitaires
```bash
pytest tests/
```

---

## Algorithmes d'optimisation

| Algorithme | Forces | Paramètres clés |
|---|---|---|
| **Génétique (GA)** | Exploration globale, robuste aux minima locaux | `pop_size=20`, `n_generations=25`, SBX + mutation polynomiale |
| **CMA-ES** | Convergence fine, adapte covariance et pas | Warm-start depuis GA, parallélisable |
| **RL (REINFORCE)** | Capture corrélations entre segments | `n_episodes=40`, décroissance de σ |

Le GA utilise le **critère 1/5 de Rechenberg** pour l'adaptation automatique du taux de mutation, et une initialisation par **Latin Hypercube Sampling (LHS)** pour une couverture uniforme de l'espace de recherche.

---

## Configuration FDTD

```python
FDTDConfig(
    nx=150, ny=150,     # Grille 150×150 cellules
    ppw=15,             # 15 points par longueur d'onde
    freq=10e9,          # 10 GHz (bande X)
    courant=0.5,        # Nombre de Courant (stabilité)
    n_steps=350,        # Pas temporels
)
```

La simulation utilise :
- **Grille de Yee** décalée (Ez aux nœuds entiers, Hx/Hy aux demi-nœuds)
- **CPML** (Convolutional PML) pour absorber les ondes sortantes
- **TFSF** pour injecter l'onde plane incidente
- **Mur PEC** paramétrique à N points de contrôle (interpolation spline)

---

## Résultats

Les figures sont sauvegardées dans `results/` :

| Fichier | Contenu |
|---|---|
| `01_baseline_snapshots.png` | Instantanés du champ Ez (mur plat) |
| `02_baseline_final.png` | État final baseline |
| `03_ga_convergence.png` | Courbe de convergence GA |
| `04_ga_profile.png` | Profil de mur optimal (GA) |
| `05_ga_field.png` | Carte de champ après GA |
| `06_rl_convergence.png` | Convergence RL |
| `07_rl_profile.png` | Profil optimal (RL) |
| `08_rl_field.png` | Carte de champ après RL |
| `09_comparison.png` | Comparaison GA / RL / baseline |
| `10_summary.png` | Résumé du projet |

---

## Références scientifiques

- **Yee, K.S.** (1966) — Algorithme FDTD original
- **Deb & Agrawal** (1995) — Simulated Binary Crossover (SBX)
- **Deb & Goyal** (1996) — Mutation polynomiale
- **Rechenberg, I.** (1973) — Critère 1/5 pour l'adaptation du pas
- **McKay et al.** (1979) — Latin Hypercube Sampling
- **Baker, J.E.** (1985) — Sélection par rang

---

## Stack technique

- **Python 3.10+**
- **NumPy ≥ 1.24** — calculs numériques
- **Matplotlib ≥ 3.7** — visualisation scientifique
- **CMA ≥ 3.3** *(optionnel)* — CMA-ES
- **SciPy ≥ 1.11** *(optionnel)* — interpolation avancée
