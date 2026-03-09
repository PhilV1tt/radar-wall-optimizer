"""
================================================================================
Module FDTD 2D TMz — Simulation de la propagation d'ondes électromagnétiques
================================================================================

Implémente l'algorithme de Yee en polarisation TMz (Ez, Hx, Hy) avec :
  - Frontière TFSF (Total-Field / Scattered-Field) pour onde plane incidente
  - Conditions aux limites absorbantes (ABC) de Mur au second ordre
  - Mur PEC paramétrique dont la géométrie est optimisable
  - Calcul de la section efficace radar (RCS) monostatique via DFT running

Théorie :
---------
Les équations de Maxwell en mode TMz (Ez, Hx, Hy) se discrétisent sur la grille
de Yee selon :

    Hx^{n+1/2}(i,j+1/2) = Hx^{n-1/2}(i,j+1/2)
                          - (Δt/μΔy) [Ez^n(i,j+1) - Ez^n(i,j)]

    Hy^{n+1/2}(i+1/2,j) = Hy^{n-1/2}(i+1/2,j)
                          + (Δt/μΔx) [Ez^n(i+1,j) - Ez^n(i,j)]

    Ez^{n+1}(i,j) = Ez^{n}(i,j)
                   + (Δt/εΔx) [Hy^{n+1/2}(i+1/2,j) - Hy^{n+1/2}(i-1/2,j)]
                   - (Δt/εΔy) [Hx^{n+1/2}(i,j+1/2) - Hx^{n+1/2}(i,j-1/2)]

L'insight fondamental de Kane Yee (1966) : les champs E et H sont évalués à
des positions spatiales et temporelles décalées, créant naturellement des
différences finies centrées d'ordre 2 en espace et en temps.

Référence : J.B. Schneider, "Understanding the FDTD Method" (ufdtd.pdf)
================================================================================
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple


# ==============================================================================
# Constantes physiques
# ==============================================================================
C0 = 299_792_458.0          # Vitesse de la lumière (m/s)
MU0 = 4.0 * np.pi * 1e-7   # Perméabilité du vide (H/m)
EPS0 = 1.0 / (MU0 * C0**2) # Permittivité du vide (F/m)
ETA0 = np.sqrt(MU0 / EPS0) # Impédance du vide ≈ 377 Ω


@dataclass
class FDTDConfig:
    """Configuration de la simulation FDTD 2D TMz.
    
    Paramètres
    ----------
    nx, ny : int
        Nombre de cellules selon x et y.
    dx : float
        Pas spatial (m). On suppose Δx = Δy = dx.
    freq : float
        Fréquence centrale du radar (Hz).
    ppw : int
        Points par longueur d'onde (contrôle la résolution).
    n_pml : int
        Épaisseur de la couche TFSF margin / ABC.
    courant : float
        Nombre de Courant Sc = c·Δt/Δx. Stabilité requiert Sc ≤ 1/√2 en 2D.
    n_steps : int
        Nombre de pas de temps.
    wall_center_x : int
        Position x du centre du mur (en cellules).
    wall_center_y : int
        Position y du centre du mur (en cellules).
    """
    # Grille
    nx: int = 200
    ny: int = 200
    ppw: int = 20        # points par longueur d'onde
    freq: float = 10e9   # 10 GHz (bande X, radar typique)
    
    # Paramètres dérivés (calculés dans __post_init__)
    dx: float = field(init=False)
    dt: float = field(init=False)
    wavelength: float = field(init=False)
    courant: float = 0.5  # Sc = c·dt/dx — bien en-dessous de 1/√2 ≈ 0.707
    n_steps: int = 500
    
    # TFSF
    tfsf_margin: int = 15  # marge en cellules autour du TFSF
    
    # Mur
    wall_center_x: int = 120
    wall_center_y: int = 100
    
    def __post_init__(self):
        self.wavelength = C0 / self.freq
        self.dx = self.wavelength / self.ppw
        self.dt = self.courant * self.dx / C0


class RickerSource:
    """Source de Ricker (dérivée seconde de Gaussienne).
    
    La wavelet de Ricker est le signal standard en simulation FDTD car :
    1. Son spectre est borné (pas de composantes DC problématiques)
    2. Elle a un contenu fréquentiel bien défini centré sur fp
    3. La fréquence maximale est environ 2.5 × fp
    
    Forme temporelle : (1 - 2(πfp·t')²) exp(-(πfp·t')²)
    où t' = t - t_delay est le temps retardé.
    """
    
    def __init__(self, fp: float, dt: float, ppw: int):
        self.fp = fp
        self.dt = dt
        self.ppw = ppw
        # Retard pour que le pulse démarre à ~0 au temps t=0
        self.t_delay = 1.0 / fp  
        
    def __call__(self, time_step: int) -> float:
        t = time_step * self.dt - self.t_delay
        arg = (np.pi * self.fp * t) ** 2
        return (1.0 - 2.0 * arg) * np.exp(-arg)
    
    def frequency_content(self, time_step: int) -> float:
        """Retourne le spectre instantané pour monitoring."""
        return self(time_step)


class FDTD2D_TMz:
    """Simulateur FDTD 2D en polarisation TMz.
    
    Polarisation TMz : Ez, Hx, Hy sont les composantes non-nulles.
    C'est le mode approprié pour étudier la diffusion par des objets 2D
    invariants selon z, comme un profil de mur.
    
    La grille de Yee en 2D TMz :
    
        Ez(i,j)       est défini aux nœuds entiers (i, j)
        Hx(i,j+1/2)  est défini à mi-chemin en y
        Hy(i+1/2,j)  est défini à mi-chemin en x
        
    Les champs E sont à des instants entiers n·Δt,
    les champs H à des instants demi-entiers (n+1/2)·Δt.
    """
    
    def __init__(self, config: FDTDConfig):
        self.cfg = config
        nx, ny = config.nx, config.ny
        
        # --- Allocation des champs ---
        # Ez : (nx, ny) — nœuds entiers
        self.Ez = np.zeros((nx, ny), dtype=np.float64)
        # Hx : (nx, ny-1) — décalé de 1/2 en y
        self.Hx = np.zeros((nx, ny - 1), dtype=np.float64)
        # Hy : (nx-1, ny) — décalé de 1/2 en x
        self.Hy = np.zeros((nx - 1, ny), dtype=np.float64)
        
        # --- Coefficients de mise à jour ---
        # Pour le vide : Chxe = Δt/(μ₀·Δx), Chye = Δt/(μ₀·Δy)
        #                Cezh = Δt/(ε₀·Δx)  (et idem en y car Δx=Δy)
        self.Chxe = config.dt / (MU0 * config.dx)  # scalaire pour milieu homogène
        self.Chye = config.dt / (MU0 * config.dx)
        self.Cezh = config.dt / (EPS0 * config.dx)
        
        # Masque PEC pour le mur : si True, Ez est forcé à 0
        self.pec_mask = np.zeros((nx, ny), dtype=bool)
        
        # --- Source ---
        self.source = RickerSource(config.freq, config.dt, config.ppw)
        
        # --- Grille 1D auxiliaire pour TFSF ---
        self._init_tfsf()
        
        # --- ABC de Mur 2nd ordre ---
        self._init_abc()
        
        # --- DFT running pour calcul de RCS ---
        self._init_dft()
        
        # Compteur de pas
        self.time_step = 0
        
    # ==========================================================================
    # Initialisation TFSF (Total-Field / Scattered-Field)
    # ==========================================================================
    def _init_tfsf(self):
        """Initialise la frontière TFSF avec grille 1D auxiliaire.
        
        Principe : On utilise une simulation 1D auxiliaire pour propager
        l'onde incidente. Cette grille 1D fournit les valeurs exactes
        du champ incident aux nœuds de la frontière TFSF.
        
        L'avantage sur l'utilisation d'expressions analytiques est que
        la dispersion numérique dans la grille 1D correspond exactement
        à celle de la grille 2D le long de l'axe de propagation.
        """
        cfg = self.cfg
        m = cfg.tfsf_margin
        
        # Limites de la région total-field
        self.tfsf_x0 = m
        self.tfsf_x1 = cfg.nx - m - 1
        self.tfsf_y0 = m
        self.tfsf_y1 = cfg.ny - m - 1
        
        # Grille 1D auxiliaire : longueur = largeur du domaine 2D + marge
        self.aux_size = cfg.nx + 2 * m
        self.ez1d = np.zeros(self.aux_size)
        self.hy1d = np.zeros(self.aux_size)
        
        # Position de la source dans la grille 1D
        self.aux_source_pos = 5
        
        # Coefficients 1D
        self.c_ez1d = cfg.courant * ETA0
        self.c_hy1d = cfg.courant / ETA0
        
    def _update_tfsf(self):
        """Met à jour la grille 1D auxiliaire et applique les corrections TFSF.
        
        Étapes :
        1. Avancer les champs 1D (onde incidente)
        2. Corriger les champs H 2D tangents à la frontière TFSF
        3. Corriger les champs E 2D tangents à la frontière TFSF
        """
        cfg = self.cfg
        
        # --- Mise à jour de la grille 1D auxiliaire ---
        # ABC simple à droite
        self.hy1d[-1] = self.hy1d[-2]
        
        # Update Hy 1D
        self.hy1d[:-1] += self.c_hy1d * (self.ez1d[1:] - self.ez1d[:-1])
        
        # Source hard dans la grille 1D
        self.hy1d[self.aux_source_pos] -= self.source(self.time_step) / ETA0
        
        # ABC simple à gauche
        self.ez1d[0] = self.ez1d[1]
        
        # Update Ez 1D
        self.ez1d[1:] += self.c_ez1d * (self.hy1d[1:] - self.hy1d[:-1])
        
        # Source additive
        self.ez1d[self.aux_source_pos + 1] += self.source(self.time_step)
        
        # --- Corrections TFSF sur la grille 2D ---
        x0, x1 = self.tfsf_x0, self.tfsf_x1
        y0, y1 = self.tfsf_y0, self.tfsf_y1
        
        # L'onde incidente se propage en +x.
        # Les corrections sont nécessaires pour les nœuds qui ont un voisin
        # de l'autre côté de la frontière TFSF.
        
        # Correction Hy sur le bord gauche (x = x0 - 1/2)
        # Hy est dans SF, son voisin Ez(x0,:) est dans TF
        self.Hy[x0 - 1, y0:y1 + 1] -= self.Chye * self.ez1d[x0]
        
        # Correction Hy sur le bord droit (x = x1 + 1/2)
        # Hy est dans SF, son voisin Ez(x1,:) est dans TF
        self.Hy[x1, y0:y1 + 1] += self.Chye * self.ez1d[x1]
        
        # Correction Hx sur le bord bas (y = y0 - 1/2)
        # Pas de correction nécessaire pour Hx car l'onde se propage en x
        # (Hx_inc = 0 pour une onde se propageant en x en TMz)
        
        # Correction Ez sur le bord gauche (x = x0)
        self.Ez[x0, y0:y1 + 1] -= self.Cezh * self.hy1d[x0 - 1]
        
        # Correction Ez sur le bord droit (x = x1)
        self.Ez[x1, y0:y1 + 1] += self.Cezh * self.hy1d[x1]
        
    # ==========================================================================
    # ABC de Mur (Absorbing Boundary Conditions)
    # ==========================================================================
    def _init_abc(self):
        """Initialise les conditions absorbantes de Mur au 1er ordre.
        
        L'ABC de Mur utilise la relation de dispersion pour estimer la
        valeur du champ sortant. Pour le 1er ordre :
        
            Ez^{n+1}(bord) = Ez^n(bord+1) + (Sc-1)/(Sc+1) × [Ez^{n+1}(bord+1) - Ez^n(bord)]
        
        où Sc est le nombre de Courant.
        """
        cfg = self.cfg
        Sc = cfg.courant
        self.abc_coeff = (Sc - 1.0) / (Sc + 1.0)
        
        # Stockage des valeurs précédentes pour l'ABC
        self.ez_prev_left = np.zeros(cfg.ny)
        self.ez_prev_right = np.zeros(cfg.ny)
        self.ez_prev_bottom = np.zeros(cfg.nx)
        self.ez_prev_top = np.zeros(cfg.nx)
        
        self.ez_prev_left_inner = np.zeros(cfg.ny)
        self.ez_prev_right_inner = np.zeros(cfg.ny)
        self.ez_prev_bottom_inner = np.zeros(cfg.nx)
        self.ez_prev_top_inner = np.zeros(cfg.nx)
        
    def _apply_abc(self):
        """Applique les ABC de Mur au 1er ordre sur les 4 bords."""
        c = self.abc_coeff
        
        # Bord gauche (x=0)
        self.Ez[0, :] = self.ez_prev_left_inner + c * (self.Ez[1, :] - self.Ez[0, :])
        self.ez_prev_left[:] = self.Ez[0, :]
        self.ez_prev_left_inner[:] = self.Ez[1, :]
        
        # Bord droit (x=nx-1)
        self.Ez[-1, :] = self.ez_prev_right_inner + c * (self.Ez[-2, :] - self.Ez[-1, :])
        self.ez_prev_right[:] = self.Ez[-1, :]
        self.ez_prev_right_inner[:] = self.Ez[-2, :]
        
        # Bord bas (y=0)
        self.Ez[:, 0] = self.ez_prev_bottom_inner + c * (self.Ez[:, 1] - self.Ez[:, 0])
        self.ez_prev_bottom[:] = self.Ez[:, 0]
        self.ez_prev_bottom_inner[:] = self.Ez[:, 1]
        
        # Bord haut (y=ny-1)
        self.Ez[:, -1] = self.ez_prev_top_inner + c * (self.Ez[:, -2] - self.Ez[:, -1])
        self.ez_prev_top[:] = self.Ez[:, -1]
        self.ez_prev_top_inner[:] = self.Ez[:, -2]
        
    # ==========================================================================
    # DFT running pour Near-to-Far-Field Transform
    # ==========================================================================
    def _init_dft(self):
        """Initialise les accumulateurs DFT pour la transformation champ proche → champ lointain.
        
        On accumule la DFT des champs tangents à une surface d'intégration
        rectangulaire entourant le diffuseur. Ces champs spectraux permettent
        ensuite de calculer la RCS via le principe d'équivalence.
        
        Pour TMz, les champs non-nuls sont Ez, Hx, Hy. Sur la surface :
        - J = n̂ × H  (courant électrique de surface)
        - M = -n̂ × E (courant magnétique de surface)
        """
        cfg = self.cfg
        m = cfg.tfsf_margin
        
        # Surface d'intégration (juste à l'intérieur de la frontière TFSF, dans le SF)
        self.dft_x0 = m + 2
        self.dft_x1 = cfg.nx - m - 3
        self.dft_y0 = m + 2
        self.dft_y1 = cfg.ny - m - 3
        
        # Nombre de nœuds sur chaque côté
        self.n_dft_x = self.dft_x1 - self.dft_x0 + 1
        self.n_dft_y = self.dft_y1 - self.dft_y0 + 1
        
        # Fréquence d'analyse
        self.omega = 2.0 * np.pi * cfg.freq
        
        # Accumulateurs DFT (complexes) pour Ez, Hx, Hy sur chaque bord
        # Bords gauche/droit : n_dft_y points
        # Bords bas/haut : n_dft_x points
        self.dft_ez_left = np.zeros(self.n_dft_y, dtype=complex)
        self.dft_ez_right = np.zeros(self.n_dft_y, dtype=complex)
        self.dft_ez_bottom = np.zeros(self.n_dft_x, dtype=complex)
        self.dft_ez_top = np.zeros(self.n_dft_x, dtype=complex)
        
        self.dft_hx_bottom = np.zeros(self.n_dft_x, dtype=complex)
        self.dft_hx_top = np.zeros(self.n_dft_x, dtype=complex)
        
        self.dft_hy_left = np.zeros(self.n_dft_y, dtype=complex)
        self.dft_hy_right = np.zeros(self.n_dft_y, dtype=complex)
        
    def _update_dft(self):
        """Accumule la DFT running à la fréquence d'analyse.
        
        Le running DFT consiste à accumuler :
            X̂(ω) = Σ_n x(n·Δt) · exp(-jω·n·Δt) · Δt
        à chaque pas de temps, évitant de stocker l'historique complet.
        """
        t_e = self.time_step * self.cfg.dt        # temps pour E
        t_h = (self.time_step + 0.5) * self.cfg.dt  # temps pour H (décalé de dt/2)
        
        phase_e = np.exp(-1j * self.omega * t_e) * self.cfg.dt
        phase_h = np.exp(-1j * self.omega * t_h) * self.cfg.dt
        
        x0, x1 = self.dft_x0, self.dft_x1
        y0, y1 = self.dft_y0, self.dft_y1
        
        # Bord gauche (x = x0) : normal n̂ = -x̂
        self.dft_ez_left += self.Ez[x0, y0:y1+1] * phase_e
        self.dft_hy_left += self.Hy[x0-1, y0:y1+1] * phase_h  # Hy juste à gauche
        
        # Bord droit (x = x1) : normal n̂ = +x̂
        self.dft_ez_right += self.Ez[x1, y0:y1+1] * phase_e
        self.dft_hy_right += self.Hy[x1, y0:y1+1] * phase_h
        
        # Bord bas (y = y0) : normal n̂ = -ŷ
        self.dft_ez_bottom += self.Ez[x0:x1+1, y0] * phase_e
        self.dft_hx_bottom += self.Hx[x0:x1+1, y0-1] * phase_h
        
        # Bord haut (y = y1) : normal n̂ = +ŷ
        self.dft_ez_top += self.Ez[x0:x1+1, y1] * phase_e
        self.dft_hx_top += self.Hx[x0:x1+1, y1] * phase_h
    
    # ==========================================================================
    # Géométrie du mur
    # ==========================================================================
    def set_wall_geometry(self, profile: np.ndarray):
        """Définit la géométrie du mur à partir d'un profil de surface.
        
        Le profil est un vecteur 1D donnant le déplacement en x de la surface
        du mur pour chaque position y. Le mur est un PEC (conducteur parfait).
        
        Paramètres
        ----------
        profile : np.ndarray
            Vecteur de taille ny donnant le décalage en x (en cellules) de la
            surface du mur par rapport à wall_center_x. Les cellules à droite
            de la surface sont remplies de PEC.
        """
        self.pec_mask[:] = False
        cx = self.cfg.wall_center_x
        
        for j in range(len(profile)):
            surface_x = cx + int(profile[j])
            # Tout ce qui est à droite de la surface est PEC
            if 0 <= surface_x < self.cfg.nx:
                self.pec_mask[surface_x:, j] = True
                
    def set_wall_from_params(self, params: np.ndarray, n_segments: int = 20, 
                              wall_height: int = 60, wall_thickness: int = 5):
        """Crée un profil de mur à partir de paramètres d'optimisation.
        
        Le profil est défini par n_segments points de contrôle qui sont
        interpolés pour donner la forme continue de la surface.
        
        Paramètres
        ----------
        params : np.ndarray
            Vecteur de n_segments valeurs dans [-1, 1], chacune contrôlant
            le déplacement latéral d'un segment du mur.
        n_segments : int
            Nombre de segments de contrôle.
        wall_height : int
            Hauteur du mur en cellules.
        wall_thickness : int
            Épaisseur de base du mur en cellules.
        """
        self.pec_mask[:] = False
        cfg = self.cfg
        cx = cfg.wall_center_x
        cy = cfg.wall_center_y
        
        # Demi-hauteur du mur
        half_h = wall_height // 2
        y_start = max(0, cy - half_h)
        y_end = min(cfg.ny, cy + half_h)
        
        # Interpolation des paramètres sur la hauteur du mur
        y_control = np.linspace(0, 1, n_segments)
        y_wall = np.linspace(0, 1, y_end - y_start)
        
        # Amplitude max du déplacement : ±10 cellules
        max_displacement = 10
        displacements = np.interp(y_wall, y_control, params) * max_displacement
        
        for idx, j in enumerate(range(y_start, y_end)):
            # Surface gauche du mur (face au radar)
            surface_x = cx + int(displacements[idx])
            # Le mur s'étend sur wall_thickness cellules
            x_start = max(0, surface_x)
            x_end = min(cfg.nx, surface_x + wall_thickness)
            if x_start < x_end:
                self.pec_mask[x_start:x_end, j] = True
                
    # ==========================================================================
    # Boucle de simulation principale
    # ==========================================================================
    def reset(self):
        """Remet tous les champs à zéro pour une nouvelle simulation."""
        self.Ez[:] = 0.0
        self.Hx[:] = 0.0
        self.Hy[:] = 0.0
        self.ez1d[:] = 0.0
        self.hy1d[:] = 0.0
        self.time_step = 0
        
        # Reset ABC
        self.ez_prev_left[:] = 0
        self.ez_prev_right[:] = 0
        self.ez_prev_bottom[:] = 0
        self.ez_prev_top[:] = 0
        self.ez_prev_left_inner[:] = 0
        self.ez_prev_right_inner[:] = 0
        self.ez_prev_bottom_inner[:] = 0
        self.ez_prev_top_inner[:] = 0
        
        # Reset DFT
        self.dft_ez_left[:] = 0
        self.dft_ez_right[:] = 0
        self.dft_ez_bottom[:] = 0
        self.dft_ez_top[:] = 0
        self.dft_hx_bottom[:] = 0
        self.dft_hx_top[:] = 0
        self.dft_hy_left[:] = 0
        self.dft_hy_right[:] = 0
    
    def step(self):
        """Avance la simulation d'un pas de temps.
        
        Ordre des opérations (crucial pour la cohérence du leap-frog) :
        1. Mise à jour de Hx et Hy  (n-1/2 → n+1/2)
        2. Corrections TFSF sur H
        3. Mise à jour de Ez         (n → n+1)
        4. Corrections TFSF sur E
        5. Application du masque PEC  (Ez = 0 sur le conducteur)
        6. Conditions absorbantes ABC
        7. Accumulation DFT
        """
        cfg = self.cfg
        
        # 1. Mise à jour des champs magnétiques
        # Hx^{n+1/2} = Hx^{n-1/2} - Chxe * (Ez[i,j+1] - Ez[i,j])
        self.Hx[:, :] -= self.Chxe * (self.Ez[:, 1:] - self.Ez[:, :-1])
        
        # Hy^{n+1/2} = Hy^{n-1/2} + Chye * (Ez[i+1,j] - Ez[i,j])
        self.Hy[:, :] += self.Chye * (self.Ez[1:, :] - self.Ez[:-1, :])
        
        # 2-4. TFSF (corrections H et E + avancement grille 1D)
        self._update_tfsf()
        
        # 3. Mise à jour du champ électrique
        # Ez^{n+1} = Ez^{n} + Cezh * ((Hy[i+1/2] - Hy[i-1/2]) - (Hx[j+1/2] - Hx[j-1/2]))
        self.Ez[1:-1, 1:-1] += self.Cezh * (
            (self.Hy[1:, 1:-1] - self.Hy[:-1, 1:-1]) -
            (self.Hx[1:-1, 1:] - self.Hx[1:-1, :-1])
        )
        
        # 5. PEC : forcer Ez = 0 dans le conducteur
        self.Ez[self.pec_mask] = 0.0
        
        # 6. ABC
        self._apply_abc()
        
        # 7. DFT running
        self._update_dft()
        
        self.time_step += 1
        
    def run(self, n_steps: Optional[int] = None) -> None:
        """Lance la simulation pour n_steps pas de temps."""
        if n_steps is None:
            n_steps = self.cfg.n_steps
        for _ in range(n_steps):
            self.step()
    
    # ==========================================================================
    # Calcul de la RCS (Radar Cross Section)
    # ==========================================================================
    def compute_rcs_backscatter(self) -> float:
        """Calcule la RCS monostatique (rétrodiffusion) via NTFF.
        
        La RCS 2D (en unités de longueur) est définie par :
        
            σ₂D(φ) = lim_{ρ→∞} 2πρ |Ez^s(ρ,φ)|² / |Ez^inc|²
        
        Pour la rétrodiffusion, φ = π (direction opposée à la propagation).
        
        La transformation NTFF utilise le principe d'équivalence :
        les courants de surface équivalents J et M sur la surface d'intégration
        sont utilisés pour calculer le champ lointain via les potentiels vecteurs.
        
        Returns
        -------
        rcs : float
            Section efficace radar 2D normalisée par la longueur d'onde (σ/λ).
            Plus cette valeur est PETITE dans la direction de rétrodiffusion,
            meilleur est le "furtivité" du mur.
        """
        cfg = self.cfg
        k = 2.0 * np.pi * cfg.freq / C0  # nombre d'onde
        dx = cfg.dx
        
        # Angle de rétrodiffusion : φ = π (retour vers la source)
        phi = np.pi
        cos_phi = np.cos(phi)  # = -1
        sin_phi = np.sin(phi)  # = 0
        
        x0, x1 = self.dft_x0, self.dft_x1
        y0, y1 = self.dft_y0, self.dft_y1
        
        # Intégrales N2D et L2D pour la transformation
        N_z = 0.0 + 0.0j   # composante z de N2D
        L_phi = 0.0 + 0.0j  # composante phi de L2D
        
        # --- Bord gauche (n̂ = -x̂) ---
        for idx, j in enumerate(range(y0, y1 + 1)):
            x_pos = x0 * dx
            y_pos = j * dx
            rho_prime_cos_psi = x_pos * cos_phi + y_pos * sin_phi
            phase = np.exp(1j * k * rho_prime_cos_psi)
            
            # J = n̂ × H = (-x̂) × (Hx·x̂ + Hy·ŷ) = -Hy·ẑ (sur le bord gauche, Hy)
            # Jz = Hy (signe: n̂ = -x̂ → Jz = +Hy pour ce bord)
            Jz = -self.dft_hy_left[idx]  # n̂ = -x̂, J = n̂ × H → Jz = -Hy
            
            # M = -n̂ × E = -(-x̂) × Ez·ẑ = +x̂ × Ez·ẑ = -Ez·ŷ
            # My = -Ez
            Mz_equiv = self.dft_ez_left[idx]  # contribution à L2D
            
            N_z += Jz * phase * dx
            L_phi += Mz_equiv * phase * dx
            
        # --- Bord droit (n̂ = +x̂) ---
        for idx, j in enumerate(range(y0, y1 + 1)):
            x_pos = x1 * dx
            y_pos = j * dx
            rho_prime_cos_psi = x_pos * cos_phi + y_pos * sin_phi
            phase = np.exp(1j * k * rho_prime_cos_psi)
            
            Jz = self.dft_hy_right[idx]
            Mz_equiv = -self.dft_ez_right[idx]
            
            N_z += Jz * phase * dx
            L_phi += Mz_equiv * phase * dx
            
        # --- Bord bas (n̂ = -ŷ) ---
        for idx, i in enumerate(range(x0, x1 + 1)):
            x_pos = i * dx
            y_pos = y0 * dx
            rho_prime_cos_psi = x_pos * cos_phi + y_pos * sin_phi
            phase = np.exp(1j * k * rho_prime_cos_psi)
            
            Jz = self.dft_hx_bottom[idx]
            Mz_equiv = self.dft_ez_bottom[idx]
            
            N_z += Jz * phase * dx
            L_phi += Mz_equiv * phase * dx
            
        # --- Bord haut (n̂ = +ŷ) ---
        for idx, i in enumerate(range(x0, x1 + 1)):
            x_pos = i * dx
            y_pos = y1 * dx
            rho_prime_cos_psi = x_pos * cos_phi + y_pos * sin_phi
            phase = np.exp(1j * k * rho_prime_cos_psi)
            
            Jz = -self.dft_hx_top[idx]
            Mz_equiv = -self.dft_ez_top[idx]
            
            N_z += Jz * phase * dx
            L_phi += Mz_equiv * phase * dx
        
        # Champ lointain Ez^s dans la direction φ
        # Ez^s ∝ ωμ₀·N_z + k·L_phi  (cf. eq. 14.54 du textbook)
        omega = 2.0 * np.pi * cfg.freq
        Ez_far = omega * MU0 * N_z + k * L_phi
        
        # RCS 2D : σ₂D = (k/(4π)) |Ez_far|² / |E_inc|²
        # On normalise par l'amplitude de la source
        # Pour la Ricker wavelet, le spectre à la fréquence centrale
        E_inc_spectrum = self._get_incident_spectrum()
        
        if abs(E_inc_spectrum) < 1e-30:
            return 1e6  # valeur de pénalité si la source n'a pas assez d'énergie
        
        rcs = (k / (4.0 * np.pi)) * abs(Ez_far)**2 / abs(E_inc_spectrum)**2
        
        # Normalisation par λ
        rcs_normalized = rcs / cfg.wavelength
        
        return rcs_normalized
    
    def _get_incident_spectrum(self) -> complex:
        """Calcule le spectre du champ incident à la fréquence d'analyse."""
        # On calcule la DFT de la source Ricker
        dt = self.cfg.dt
        omega = self.omega
        spectrum = 0.0 + 0.0j
        for n in range(self.time_step):
            spectrum += self.source(n) * np.exp(-1j * omega * n * dt) * dt
        return spectrum
    
    def compute_backscatter_energy(self) -> float:
        """Méthode simplifiée : calcule l'énergie du champ diffusé dans la 
        direction de rétrodiffusion.
        
        Plus simple et plus robuste que la NTFF complète. On mesure l'énergie
        du champ Ez dans la zone scattered-field, côté source (à gauche du mur).
        
        C'est l'approche recommandée pour l'optimisation car elle est plus
        stable numériquement.
        """
        # Zone de mesure : région SF à gauche du mur
        m = self.cfg.tfsf_margin
        measure_x = slice(m + 2, self.cfg.wall_center_x - 20)
        measure_y = slice(m + 2, self.cfg.ny - m - 2)
        
        # Énergie du champ diffusé
        scattered_energy = np.sum(self.Ez[measure_x, measure_y]**2)
        
        return scattered_energy
    
    def compute_bistatic_rcs(self, n_angles: int = 360) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule la RCS bistatique pour n_angles directions.
        
        Retourne le diagramme de diffusion complet σ(φ) pour φ ∈ [0, 2π].
        Utile pour visualiser le pattern de diffusion.
        """
        cfg = self.cfg
        k = 2.0 * np.pi * cfg.freq / C0
        dx = cfg.dx
        omega = 2.0 * np.pi * cfg.freq
        
        angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        rcs = np.zeros(n_angles)
        
        E_inc = self._get_incident_spectrum()
        if abs(E_inc) < 1e-30:
            return angles, np.ones(n_angles) * 1e6
        
        x0, x1 = self.dft_x0, self.dft_x1
        y0, y1 = self.dft_y0, self.dft_y1
        
        for a_idx, phi in enumerate(angles):
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            N_z = 0.0 + 0.0j
            L_z = 0.0 + 0.0j
            
            # Bord gauche
            for idx, j in enumerate(range(y0, y1+1)):
                rpc = x0 * dx * cos_phi + j * dx * sin_phi
                phase = np.exp(1j * k * rpc)
                N_z += (-self.dft_hy_left[idx]) * phase * dx
                L_z += self.dft_ez_left[idx] * phase * dx
                
            # Bord droit
            for idx, j in enumerate(range(y0, y1+1)):
                rpc = x1 * dx * cos_phi + j * dx * sin_phi
                phase = np.exp(1j * k * rpc)
                N_z += self.dft_hy_right[idx] * phase * dx
                L_z += (-self.dft_ez_right[idx]) * phase * dx
                
            # Bord bas
            for idx, i in enumerate(range(x0, x1+1)):
                rpc = i * dx * cos_phi + y0 * dx * sin_phi
                phase = np.exp(1j * k * rpc)
                N_z += self.dft_hx_bottom[idx] * phase * dx
                L_z += self.dft_ez_bottom[idx] * phase * dx
                
            # Bord haut
            for idx, i in enumerate(range(x0, x1+1)):
                rpc = i * dx * cos_phi + y1 * dx * sin_phi
                phase = np.exp(1j * k * rpc)
                N_z += (-self.dft_hx_top[idx]) * phase * dx
                L_z += (-self.dft_ez_top[idx]) * phase * dx
            
            Ez_far = omega * MU0 * N_z + k * L_z
            rcs[a_idx] = (k / (4*np.pi)) * abs(Ez_far)**2 / abs(E_inc)**2
            
        return angles, rcs / cfg.wavelength
